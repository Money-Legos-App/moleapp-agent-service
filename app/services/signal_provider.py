"""
Signal Provider (The Brain)

Runs once per trading cycle to produce a MarketState:
- Fetches market data from Hyperliquid
- Queries FAISS for pattern context (optional)
- Calls DeepSeek LLM concurrently for all assets
- Stores the resulting MarketState in Redis

This is the "One Brain" in the "One Brain, Many Hands" architecture.
The Execution Fleet (dispatcher + workers) consumes the MarketState.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog

from app.config import get_settings

logger = structlog.get_logger(__name__)

# Max concurrent LLM calls to avoid rate-limiting
MAX_CONCURRENT_ANALYSES = 5


class MarketState:
    """
    Immutable snapshot of market conditions + LLM signals.
    Produced once per cycle by the Signal Provider, consumed by N workers.
    """

    def __init__(
        self,
        cycle_id: str,
        triggered_at: str,
        signals: List[Dict[str, Any]],
        market_data: Dict[str, Dict[str, Any]],
        risk_metrics: Dict[str, Dict[str, Any]],
        errors: List[str],
    ):
        self.cycle_id = cycle_id
        self.triggered_at = triggered_at
        self.signals = signals
        self.market_data = market_data
        self.risk_metrics = risk_metrics
        self.errors = errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "triggered_at": self.triggered_at,
            "signals": self.signals,
            "market_data": self.market_data,
            "risk_metrics": self.risk_metrics,
            "errors": self.errors,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarketState":
        return cls(
            cycle_id=data["cycle_id"],
            triggered_at=data["triggered_at"],
            signals=data["signals"],
            market_data=data["market_data"],
            risk_metrics=data["risk_metrics"],
            errors=data.get("errors", []),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_json(cls, raw: str) -> "MarketState":
        return cls.from_dict(json.loads(raw))


async def generate_market_state(trigger_type: str = "scheduled") -> MarketState:
    """
    Run market analysis and produce a MarketState.

    This is the core "Brain" function. It:
    1. Fetches market data from Hyperliquid for all allowed assets
    2. Queries FAISS for historical patterns (graceful fallback)
    3. Calls DeepSeek LLM for signal generation
    4. Returns a MarketState object

    Args:
        trigger_type: What triggered this analysis cycle

    Returns:
        MarketState with signals and market context
    """
    from app.services.llm import DeepSeekClient
    from app.services.hyperliquid import HyperliquidClient
    from app.services.observability.langfuse_client import get_langfuse
    from app.services.observability.prompt_manager import get_prompt_manager

    settings = get_settings()
    cycle_id = f"cycle_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    logger.info(
        "Signal Provider: starting market analysis",
        cycle_id=cycle_id,
        trigger=trigger_type,
    )

    start_time = time.monotonic()

    # Langfuse: root trace for the brain cycle
    lf = get_langfuse()
    pm = get_prompt_manager()
    trace = lf.start_trace(
        name="trading-cycle-brain",
        session_id=cycle_id,
        tags=[trigger_type, settings.environment],
        metadata={"cycle_id": cycle_id, "assets": settings.allowed_assets},
    )

    # FAISS/RAG: skip entirely when disabled (saves ~12s model load per cycle)
    faiss_store = None
    if not settings.disable_rag:
        try:
            from app.services.rag import FAISSStore

            faiss_store = FAISSStore()
            await faiss_store.initialize()
            if not getattr(faiss_store, "is_ready", False):
                faiss_store = None
        except Exception as e:
            logger.warning("FAISS not available, proceeding without RAG", error=str(e))

    llm = DeepSeekClient()
    hl_client = HyperliquidClient()

    signals: List[Dict[str, Any]] = []
    market_data: Dict[str, Dict[str, Any]] = {}
    risk_metrics: Dict[str, Dict[str, Any]] = {}
    errors: List[str] = []

    # Fetch ALL market data in 1 metaAndAssetCtxs call + N concurrent L2 calls
    # instead of 2N calls (1 meta + 1 L2 per asset sequentially)
    try:
        market_data = await hl_client.get_bulk_market_data(settings.allowed_assets)
    except Exception as e:
        logger.error("Failed to fetch bulk market data", error=str(e))
        errors.append(f"Bulk market data fetch failed: {str(e)}")
        await hl_client.close()
        await llm.close()
        return MarketState(
            cycle_id=cycle_id,
            triggered_at=datetime.utcnow().isoformat(),
            signals=[], market_data={}, risk_metrics={}, errors=errors,
        )

    # Fetch multi-timeframe candle data + OI delta from Redis
    from app.services.execution_queue import get_redis
    try:
        redis = await get_redis()
    except Exception:
        redis = None

    # Multi-timeframe technical analysis (concurrent for all assets)
    tf_summaries: Dict[str, str] = {}
    try:
        async def _get_tf(asset: str):
            coin = asset.replace("-USD", "")
            return asset, await hl_client.get_multi_timeframe_analysis(coin)

        tf_results = await asyncio.gather(
            *[_get_tf(a) for a in settings.allowed_assets],
            return_exceptions=True,
        )
        for result in tf_results:
            if isinstance(result, tuple):
                tf_summaries[result[0]] = result[1]
    except Exception as e:
        logger.warning("Multi-timeframe fetch failed", error=str(e))

    # OI delta + relative volume: compare current values against last cycle (Redis)
    oi_deltas: Dict[str, Dict[str, Any]] = {}
    for asset, data in market_data.items():
        coin = asset.replace("-USD", "")
        current_oi = data.get("open_interest", 0)
        current_vol = data.get("volume_24h", 0)

        delta_info: Dict[str, Any] = {"oi_change_pct": None, "vol_vs_avg": None}
        if redis:
            try:
                prev_raw = await redis.hget("agent:market:prev_oi", coin)
                prev_vol_raw = await redis.hget("agent:market:prev_vol_24h", coin)

                if prev_raw:
                    prev_oi = float(prev_raw)
                    if prev_oi > 0:
                        delta_info["oi_change_pct"] = round(
                            (current_oi - prev_oi) / prev_oi * 100, 2
                        )

                if prev_vol_raw:
                    prev_vol = float(prev_vol_raw)
                    if prev_vol > 0:
                        delta_info["vol_vs_avg"] = round(current_vol / prev_vol, 2)

                # Store current values for next cycle
                await redis.hset("agent:market:prev_oi", coin, str(current_oi))
                await redis.hset("agent:market:prev_vol_24h", coin, str(current_vol))
            except Exception:
                pass

        oi_deltas[asset] = delta_info
        # Attach to market_data for downstream consumers
        market_data[asset]["oi_change_pct"] = delta_info.get("oi_change_pct")
        market_data[asset]["vol_vs_avg"] = delta_info.get("vol_vs_avg")

    # Semaphore to limit concurrent DeepSeek API calls
    sem = asyncio.Semaphore(MAX_CONCURRENT_ANALYSES)

    async def _analyze_asset(asset: str) -> Optional[Dict[str, Any]]:
        """Analyze a single asset. Returns a signal dict or None."""
        async with sem:
            try:
                data = market_data.get(asset, {})

                current_price = data.get("price", 0)
                if current_price == 0:
                    logger.warning("No price data for asset", asset=asset)
                    return None

                # Query FAISS for patterns (wrapped in Langfuse span)
                patterns = []
                if faiss_store is not None:
                    query = (
                        f"Current {asset} market conditions: "
                        f"Price: ${current_price:g}, "
                        f"24h change: {data.get('price_change_24h', 0):+.2f}%, "
                        f"Volume: ${data.get('volume_24h', 0):,.0f}, "
                        f"Funding rate: {data.get('funding_rate', 0):.4f}%"
                    )
                    rag_span = lf.start_span(trace, name=f"rag-retrieval-{asset}")
                    rag_span.update(input={"query": query, "asset": asset, "k": 5})
                    patterns = await faiss_store.search(query=query, k=5, filters={"asset": asset})
                    rag_span.update(output={
                        "pattern_count": len(patterns),
                        "pattern_ids": [p.get("id") for p in patterns],
                    })
                    rag_span.end()

                # Build pattern context — skip dead data when RAG is disabled
                if patterns:
                    context_parts = []
                    for p in patterns:
                        outcome = p.get("outcome", {})
                        context_parts.append(
                            f"Pattern from {p.get('date', 'unknown')}: "
                            f"{p.get('context', '')} "
                            f"Outcome: {outcome.get('return_next_30d', 0)*100:+.1f}% return, "
                            f"Max drawdown: {outcome.get('max_drawdown_next_30d', 0)*100:.1f}%, "
                            f"Similarity: {p.get('similarity_score', 0):.2f}"
                        )
                    pattern_context = "\n\n".join(context_parts)
                else:
                    # When RAG is disabled, pass None to skip the section entirely
                    pattern_context = None

                # Risk context
                if faiss_store is not None:
                    risk = await faiss_store.get_risk_context(asset)
                else:
                    risk = None  # Skip hardcoded dummy values when RAG is disabled
                if risk:
                    risk_metrics[asset] = risk

                # Gather enrichment data for this asset
                tf_summary = tf_summaries.get(asset)
                oi_delta = oi_deltas.get(asset, {})
                bid_imbalance = data.get("bid_imbalance_pct", 0)

                # Fetch prompt from Langfuse (falls back to local PromptTemplates)
                lf_prompt, _ = pm.get_market_analysis_prompt(
                    asset=asset,
                    current_price=current_price,
                    price_change_24h=data.get("price_change_24h", 0),
                    volume_24h=data.get("volume_24h", 0),
                    spread=data.get("spread", 0.01),
                    pattern_context=pattern_context or "",
                    risk_metrics=risk or {},
                    funding_rate=data.get("funding_rate", 0),
                    open_interest=data.get("open_interest", 0),
                    tf_summary=tf_summary,
                    oi_delta=oi_delta,
                    bid_imbalance_pct=bid_imbalance,
                )

                # Call DeepSeek for analysis (with Langfuse trace)
                signal_response = await llm.analyze_market(
                    asset=asset,
                    current_price=current_price,
                    price_change_24h=data.get("price_change_24h", 0),
                    volume_24h=data.get("volume_24h", 0),
                    spread=data.get("spread", 0.01),
                    pattern_context=pattern_context,
                    risk_metrics=risk,
                    funding_rate=data.get("funding_rate", 0),
                    open_interest=data.get("open_interest", 0),
                    tf_summary=tf_summary,
                    oi_delta=oi_delta,
                    bid_imbalance_pct=bid_imbalance,
                    trace=trace,
                    lf_prompt=lf_prompt,
                )

                if signal_response.get("should_trade", False):
                    # ── Funding Trap Filter (hard-coded safety net) ──
                    HIGH_FUNDING_THRESHOLD = 0.0005
                    funding = data.get("funding_rate", 0)
                    direction = signal_response.get("direction")
                    confidence = signal_response.get("confidence", "LOW")

                    if direction == "LONG" and funding > HIGH_FUNDING_THRESHOLD and confidence != "HIGH":
                        logger.info(
                            "Signal REJECTED by funding trap filter",
                            asset=asset,
                            direction=direction,
                            funding_hourly_pct=f"{funding * 100:.4f}%",
                            confidence=confidence,
                            reason="LONG against high positive funding, confidence not HIGH",
                        )
                        return None

                    if direction == "SHORT" and funding < -HIGH_FUNDING_THRESHOLD and confidence != "HIGH":
                        logger.info(
                            "Signal REJECTED by funding trap filter",
                            asset=asset,
                            direction=direction,
                            funding_hourly_pct=f"{funding * 100:.4f}%",
                            confidence=confidence,
                            reason="SHORT against high negative funding, confidence not HIGH",
                        )
                        return None

                    signal = {
                        "asset": asset,
                        "direction": signal_response.get("direction", "LONG"),
                        "confidence": signal_response.get("confidence", "LOW"),
                        "recommended_leverage": signal_response.get("recommended_leverage", 1),
                        "strategy_tag": signal_response.get("strategy_tag", "unknown"),
                        "reasoning": signal_response.get("reasoning", ""),
                        "entry_zone": signal_response.get(
                            "entry_zone",
                            {"min": current_price * 0.99, "max": current_price * 1.01},
                        ),
                        "stop_loss_percent": signal_response.get("stop_loss_percent", 5),
                        "take_profit_percent": signal_response.get("take_profit_percent", 15),
                        # HL-aligned exit parameters
                        "max_hold_hours": signal_response.get("max_hold_hours", 72),
                        "trailing_stop": signal_response.get("trailing_stop", None),
                        "funding_exit_threshold": signal_response.get("funding_exit_threshold", None),
                        "thesis_invalidation": signal_response.get("thesis_invalidation", ""),
                        # Legacy field (kept for backwards compat)
                        "time_horizon": signal_response.get("time_horizon", "3d"),
                        "rag_context_ids": [p.get("id") for p in patterns] if patterns else [],
                        "max_drawdown_30d": risk.get("max_drawdown_30d"),
                        "volatility_score": risk.get("volatility_30d"),
                        "generated_at": datetime.utcnow().isoformat(),
                    }

                    # Persist signal to DB for backtesting and API
                    try:
                        from app.services.database import save_signal
                        signal_id = await save_signal(signal, cycle_id)
                        signal["signal_id"] = signal_id
                    except Exception as persist_err:
                        logger.warning("Failed to persist signal", error=str(persist_err))

                    logger.info(
                        "Signal generated",
                        cycle_id=cycle_id,
                        asset=asset,
                        direction=signal["direction"],
                        confidence=signal["confidence"],
                    )
                    return signal
                else:
                    logger.info(
                        "No trade signal",
                        asset=asset,
                        reason=signal_response.get("reasoning", ""),
                    )
                    return None

            except Exception as e:
                logger.error("Error analyzing asset", asset=asset, error=str(e))
                errors.append(f"Market analysis error for {asset}: {str(e)}")
                return None

    try:
        # Run all asset analyses concurrently (bounded by semaphore)
        results = await asyncio.gather(
            *[_analyze_asset(asset) for asset in settings.allowed_assets],
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, Exception):
                errors.append(f"Unexpected analysis error: {str(result)}")
            elif result is not None:
                signals.append(result)

    finally:
        await hl_client.close()
        await llm.close()

    duration = time.monotonic() - start_time

    logger.info(
        "Signal Provider: analysis complete",
        cycle_id=cycle_id,
        signals_count=len(signals),
        assets_analyzed=len(market_data),
        duration_seconds=round(duration, 2),
    )

    # Langfuse: update trace with final outcome
    trace.update(
        output={
            "signals_count": len(signals),
            "assets_analyzed": len(market_data),
            "errors_count": len(errors),
            "duration_seconds": round(duration, 2),
        },
    )
    trace.end()

    return MarketState(
        cycle_id=cycle_id,
        triggered_at=datetime.utcnow().isoformat(),
        signals=signals,
        market_data=market_data,
        risk_metrics=risk_metrics,
        errors=errors,
    )

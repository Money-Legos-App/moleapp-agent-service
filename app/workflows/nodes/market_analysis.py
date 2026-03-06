"""
Layer A: Market Analysis Node
Fetches market data, queries FAISS, and generates trading signals via DeepSeek
"""

from datetime import datetime
from typing import Any, Dict

import structlog

from app.workflows.state import AgentState, StrategySignal

logger = structlog.get_logger(__name__)


async def market_analysis_node(state: AgentState) -> Dict[str, Any]:
    """
    Market Analysis Node (Layer A).

    Steps:
    1. Fetch current market data for all allowed assets
    2. Query FAISS for similar historical patterns
    3. Call DeepSeek for signal generation
    4. Return trading signals

    Args:
        state: Current workflow state

    Returns:
        Updated state fields
    """
    from app.config import get_settings
    from app.services.llm import DeepSeekClient
    from app.services.hyperliquid import HyperliquidClient

    settings = get_settings()
    logger.info("Starting market analysis", trigger=state["trigger_type"])

    # Initialize services
    # FAISS/RAG is optional -- graceful fallback when not available
    faiss_store = None
    try:
        from app.services.rag import FAISSStore
        faiss_store = FAISSStore()
        await faiss_store.initialize()
        if not getattr(faiss_store, 'is_ready', False):
            logger.info("FAISS index not ready, proceeding without RAG")
            faiss_store = None
    except Exception as e:
        logger.warning("FAISS not available, proceeding without RAG", error=str(e))
        faiss_store = None

    llm = DeepSeekClient()
    hl_client = HyperliquidClient()

    signals = []
    market_data = {}
    pattern_contexts = {}
    risk_metrics = {}
    rag_context_ids = []
    errors = state.get("errors", [])

    try:
        for asset in settings.allowed_assets:
            logger.debug("Analyzing asset", asset=asset)

            try:
                # Step 1: Fetch current market data
                data = await hl_client.get_market_data(asset)
                market_data[asset] = data

                current_price = data.get("price", 0)
                if current_price == 0:
                    logger.warning("No price data for asset", asset=asset)
                    continue

                # Step 2: Query FAISS for similar patterns (if available)
                patterns = []
                if faiss_store is not None:
                    query = f"""
                    Current {asset} market conditions:
                    Price: ${current_price:,.2f}
                    24h change: {data.get('price_change_24h', 0):+.2f}%
                    Volume: ${data.get('volume_24h', 0):,.0f}
                    Funding rate: {data.get('funding_rate', 0):.4f}%
                    """

                    patterns = await faiss_store.search(
                        query=query,
                        k=5,
                        filters={"asset": asset},
                    )

                # Build pattern context string for LLM
                if patterns:
                    context_parts = []
                    for p in patterns:
                        context_parts.append(f"""
Pattern from {p.get('date', 'unknown')}:
{p.get('context', '')}
Outcome: {p.get('outcome', {}).get('return_next_30d', 0)*100:+.1f}% return over next 30 days
Max drawdown: {p.get('outcome', {}).get('max_drawdown_next_30d', 0)*100:.1f}%
Similarity: {p.get('similarity_score', 0):.2f}
                        """.strip())
                        rag_context_ids.append(p.get("id"))

                    pattern_context = "\n\n".join(context_parts)
                else:
                    pattern_context = "No historical pattern data available."

                pattern_contexts[asset] = pattern_context

                # Step 3: Get risk context (FAISS or defaults)
                if faiss_store is not None:
                    risk = await faiss_store.get_risk_context(asset)
                else:
                    risk = {"max_drawdown_30d": -0.20, "volatility_30d": 0.05, "sample_count": 0}
                risk_metrics[asset] = risk

                # Step 4: Call DeepSeek for analysis
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
                )

                # Application-level audit log
                logger.info(
                    "AUDIT market_analysis LLM decision",
                    asset=asset,
                    should_trade=signal_response.get("should_trade"),
                    direction=signal_response.get("direction"),
                    confidence=signal_response.get("confidence"),
                    reasoning=signal_response.get("reasoning", "")[:200],
                    tokens_used=signal_response.get("_audit_tokens"),
                    current_price=current_price,
                )

                # DB audit log
                await _audit_log(
                    node="market_analysis",
                    action="llm_call",
                    asset=asset,
                    llm_prompt=signal_response.get("_audit_prompt"),
                    llm_response=signal_response.get("_audit_response"),
                    llm_model=signal_response.get("_audit_model"),
                    llm_tokens=signal_response.get("_audit_tokens"),
                    decision={
                        "should_trade": signal_response.get("should_trade"),
                        "direction": signal_response.get("direction"),
                        "confidence": signal_response.get("confidence"),
                    },
                    reasoning=signal_response.get("reasoning", ""),
                    metadata={"current_price": current_price, "price_change_24h": data.get("price_change_24h", 0)},
                    success="_audit_error" not in signal_response,
                    error_message=signal_response.get("_audit_error"),
                )

                # Only include signals where trading is recommended
                if signal_response.get("should_trade", False):
                    # ── Funding Trap Filter (hard-coded safety net) ──
                    # LLMs can hallucinate; math does not.
                    # 0.0005 = 0.05% per hour ≈ 44% APR — extreme carry cost.
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
                        )
                        await _audit_log(
                            node="market_analysis",
                            action="signal_rejected_funding_trap",
                            asset=asset,
                            reasoning=f"LONG against {funding * 100:.4f}% hourly funding, confidence={confidence}",
                        )
                        continue

                    if direction == "SHORT" and funding < -HIGH_FUNDING_THRESHOLD and confidence != "HIGH":
                        logger.info(
                            "Signal REJECTED by funding trap filter",
                            asset=asset,
                            direction=direction,
                            funding_hourly_pct=f"{funding * 100:.4f}%",
                            confidence=confidence,
                        )
                        await _audit_log(
                            node="market_analysis",
                            action="signal_rejected_funding_trap",
                            asset=asset,
                            reasoning=f"SHORT against {funding * 100:.4f}% hourly funding, confidence={confidence}",
                        )
                        continue

                    signal: StrategySignal = {
                        "asset": asset,
                        "direction": signal_response.get("direction", "LONG"),
                        "confidence": signal_response.get("confidence", "LOW"),
                        "recommended_leverage": signal_response.get("recommended_leverage", 1),
                        "strategy_tag": signal_response.get("strategy_tag", "unknown"),
                        "reasoning": signal_response.get("reasoning", ""),
                        "entry_zone": signal_response.get("entry_zone", {"min": current_price * 0.99, "max": current_price * 1.01}),
                        "stop_loss_percent": signal_response.get("stop_loss_percent", 5),
                        "take_profit_percent": signal_response.get("take_profit_percent", 15),
                        "time_horizon": signal_response.get("time_horizon", "3d"),
                        "rag_context_ids": [p.get("id") for p in patterns] if patterns else [],
                        "max_drawdown_30d": risk.get("max_drawdown_30d"),
                        "volatility_score": risk.get("volatility_30d"),
                        "generated_at": datetime.utcnow(),
                    }
                    signals.append(signal)

                    # Audit: Log signal generated
                    await _audit_log(
                        node="market_analysis",
                        action="signal_generated",
                        asset=asset,
                        decision={
                            "direction": signal["direction"],
                            "confidence": signal["confidence"],
                            "leverage": signal["recommended_leverage"],
                            "strategy_tag": signal["strategy_tag"],
                        },
                        reasoning=signal["reasoning"],
                    )

                    logger.info(
                        "Signal generated",
                        asset=asset,
                        direction=signal["direction"],
                        confidence=signal["confidence"],
                        leverage=signal["recommended_leverage"],
                    )
                else:
                    # Audit: Log signal skipped
                    await _audit_log(
                        node="market_analysis",
                        action="signal_skipped",
                        asset=asset,
                        reasoning=signal_response.get("reasoning", "No trade recommended"),
                    )

                    logger.info(
                        "No trade signal",
                        asset=asset,
                        reason=signal_response.get("reasoning", ""),
                    )

            except Exception as e:
                logger.error(
                    "Error analyzing asset",
                    asset=asset,
                    error=str(e),
                )
                errors.append(f"Market analysis error for {asset}: {str(e)}")

    finally:
        await hl_client.close()
        await llm.close()

    logger.info(
        "Market analysis completed",
        signals_count=len(signals),
        assets_analyzed=len(market_data),
    )

    return {
        "signals": signals,
        "market_data": market_data,
        "pattern_contexts": pattern_contexts,
        "risk_metrics": risk_metrics,
        "rag_context_ids": list(set(rag_context_ids)),
        "errors": errors,
        "completed_nodes": state.get("completed_nodes", []) + ["market_analysis"],
    }


async def _audit_log(**kwargs) -> None:
    """Write an audit log entry, swallowing errors to avoid disrupting the pipeline."""
    try:
        from app.services.database import record_agent_audit
        await record_agent_audit(**kwargs)
    except Exception as e:
        logger.warning("Failed to write audit log", error=str(e))

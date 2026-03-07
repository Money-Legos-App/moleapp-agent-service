"""
Prompt Manager — Langfuse-first prompt system with local fallback.

Langfuse is the primary prompt source. All prompt templates are managed
in Langfuse Cloud (Mustache {{variable}} syntax). If Langfuse is unavailable
or compile() fails, the local PromptTemplates helpers are used as fallback.

Usage:
    pm = get_prompt_manager()
    lf_prompt, text = pm.get_market_analysis_prompt(
        asset="ETH-USD", current_price=3000, ...
    )
    # Pass lf_prompt to DeepSeekClient so generations are linked to prompt version
"""

from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import structlog

from app.services.llm.prompts import PromptTemplates
from app.services.observability.langfuse_client import get_langfuse

logger = structlog.get_logger(__name__)

# Langfuse prompt names — must match what's registered in Langfuse Cloud
PROMPT_NAMES = {
    "system": "trading-system-prompt",
    "market_analysis": "market-analysis-prompt",
    "user_filter": "user-filter-prompt",
    "position_exit": "position-exit-prompt",
}


class PromptManager:
    """
    Langfuse-first prompt manager.

    Primary path: Fetch prompt from Langfuse → compile() with pre-formatted strings.
    Fallback path: Local PromptTemplates helpers (Python str.format with specifiers).

    All numeric values are pre-formatted as strings before passing to Langfuse
    compile(), which does plain string substitution (no format specifiers).
    """

    def __init__(self):
        self._lf = get_langfuse()

    def _fetch(self, key: str, fallback_text: str) -> Tuple[Optional[Any], Optional[str]]:
        """
        Fetch a prompt object from Langfuse.
        Returns (lf_prompt_obj, raw_template_text).
        lf_prompt_obj is None when Langfuse is disabled or fetch fails.
        """
        lf_prompt, text = self._lf.get_prompt(
            prompt_name=PROMPT_NAMES[key],
            fallback=fallback_text,
        )
        return lf_prompt, text

    # ── System Prompt ──────────────────────────────────────────────────

    def get_system_prompt(self) -> Tuple[Optional[Any], str]:
        """Fetch the system prompt (no formatting needed)."""
        lf_prompt, text = self._fetch("system", PromptTemplates.SYSTEM_PROMPT)
        if text is None:
            text = PromptTemplates.SYSTEM_PROMPT
        return lf_prompt, text

    # ── Market Analysis Prompt ─────────────────────────────────────────

    def get_market_analysis_prompt(
        self,
        asset: str,
        current_price: float,
        price_change_24h: float,
        volume_24h: float,
        spread: float,
        pattern_context: str,
        risk_metrics: Dict[str, Any],
        funding_rate: float = 0.0,
        open_interest: float = 0.0,
        tf_summary: Optional[str] = None,
        oi_delta: Optional[Dict[str, Any]] = None,
        bid_imbalance_pct: float = 0.0,
    ) -> Tuple[Optional[Any], str]:
        """
        Fetch and format the market analysis prompt.

        Primary: Langfuse compile() with pre-formatted Mustache variables.
        Fallback: PromptTemplates.format_market_analysis() local helper.
        """
        lf_prompt, _ = self._fetch(
            "market_analysis", PromptTemplates.MARKET_ANALYSIS_PROMPT
        )

        if lf_prompt is not None:
            try:
                lf_vars = self._build_market_analysis_vars(
                    asset=asset,
                    current_price=current_price,
                    price_change_24h=price_change_24h,
                    volume_24h=volume_24h,
                    spread=spread,
                    pattern_context=pattern_context,
                    risk_metrics=risk_metrics,
                    funding_rate=funding_rate,
                    open_interest=open_interest,
                    tf_summary=tf_summary,
                    oi_delta=oi_delta,
                    bid_imbalance_pct=bid_imbalance_pct,
                )
                text = lf_prompt.compile(**lf_vars)
                logger.debug("Market analysis prompt compiled from Langfuse", asset=asset)
                return lf_prompt, text
            except Exception as e:
                logger.warning(
                    "Langfuse compile failed for market_analysis, using local fallback",
                    error=str(e),
                )

        # Fallback: local helper (handles all enrichment sections)
        text = PromptTemplates.format_market_analysis(
            asset=asset,
            current_price=current_price,
            price_change_24h=price_change_24h,
            volume_24h=volume_24h,
            spread=spread,
            pattern_context=pattern_context,
            risk_metrics=risk_metrics,
            funding_rate=funding_rate,
            open_interest=open_interest,
            tf_summary=tf_summary,
            oi_delta=oi_delta,
            bid_imbalance_pct=bid_imbalance_pct,
        )
        return lf_prompt, text

    def _build_market_analysis_vars(
        self,
        asset: str,
        current_price: float,
        price_change_24h: float,
        volume_24h: float,
        spread: float,
        pattern_context: str,
        risk_metrics: Dict[str, Any],
        funding_rate: float,
        open_interest: float,
        tf_summary: Optional[str],
        oi_delta: Optional[Dict[str, Any]],
        bid_imbalance_pct: float,
    ) -> Dict[str, str]:
        """Build pre-formatted Mustache variables for market analysis."""
        funding_rate_hourly = funding_rate * 100
        funding_rate_apr = funding_rate * 100 * 8760

        # Smart price formatting
        if current_price >= 1:
            formatted_price = f"{current_price:,.2f}"
        elif current_price >= 0.01:
            formatted_price = f"{current_price:.4f}"
        elif current_price >= 0.0001:
            formatted_price = f"{current_price:.6f}"
        else:
            formatted_price = f"{current_price:.8f}"

        # Build OI delta section
        oi_delta = oi_delta or {}
        oi_parts = []
        if oi_delta.get("oi_change_pct") is not None:
            oi_parts.append(f"- OI Change (vs last cycle): {oi_delta['oi_change_pct']:+.2f}%")
        if oi_delta.get("vol_vs_avg") is not None:
            oi_parts.append(f"- Volume vs Previous Cycle: {oi_delta['vol_vs_avg']}x")
        oi_delta_section = "\n".join(oi_parts)

        # Build orderbook section
        if bid_imbalance_pct != 0:
            if bid_imbalance_pct > 20:
                ob_label = "strong bid support"
            elif bid_imbalance_pct > 5:
                ob_label = "mild bid support"
            elif bid_imbalance_pct < -20:
                ob_label = "strong ask pressure"
            elif bid_imbalance_pct < -5:
                ob_label = "mild ask pressure"
            else:
                ob_label = "balanced"
            orderbook_section = f"- Orderbook Imbalance (5-level): {bid_imbalance_pct:+.1f}% ({ob_label})"
        else:
            orderbook_section = ""

        # Build multi-timeframe section
        tf_analysis_section = tf_summary or "Multi-timeframe data unavailable for this cycle."

        # Build pattern section
        if pattern_context:
            risk = risk_metrics or {}
            pattern_section = (
                f"\n## Historical Pattern Context (from similar market conditions)\n"
                f"{pattern_context}\n\n"
                f"## Risk Metrics from Historical Patterns\n"
                f"- Maximum Drawdown (30d): {risk.get('max_drawdown_30d', -0.20) * 100:.1f}%\n"
                f"- Average Volatility (30d): {risk.get('volatility_30d', 0.05) * 100:.1f}%\n"
                f"- Sample Size: {risk.get('sample_count', 0)} patterns\n"
            )
        else:
            pattern_section = ""

        return {
            "asset": asset,
            "current_price": formatted_price,
            "price_change_24h": f"{price_change_24h:+.2f}",
            "volume_24h": f"{volume_24h:,.0f}",
            "spread": f"{spread:.4f}",
            "funding_rate_hourly": f"{funding_rate_hourly:.4f}",
            "funding_rate_apr": f"{funding_rate_apr:.1f}",
            "open_interest": f"{open_interest:,.0f}",
            "oi_delta_section": oi_delta_section,
            "orderbook_section": orderbook_section,
            "tf_analysis_section": tf_analysis_section,
            "pattern_section": pattern_section,
        }

    # ── User Filter Prompt ─────────────────────────────────────────────

    def get_user_filter_prompt(
        self,
        signal: Dict[str, Any],
        mission: Dict[str, Any],
        existing_positions: List[Dict[str, Any]],
        margin_used: float = 0.0,
        account_value: float = 0.0,
    ) -> Tuple[Optional[Any], str]:
        """
        Fetch and format the user filter prompt.

        Primary: Langfuse compile() with pre-formatted Mustache variables.
        Fallback: PromptTemplates.format_user_filter() local helper.
        """
        lf_prompt, _ = self._fetch(
            "user_filter", PromptTemplates.USER_FILTER_PROMPT
        )

        if lf_prompt is not None:
            try:
                lf_vars = self._build_user_filter_vars(
                    signal=signal,
                    mission=mission,
                    existing_positions=existing_positions,
                    margin_used=margin_used,
                    account_value=account_value,
                )
                text = lf_prompt.compile(**lf_vars)
                logger.debug("User filter prompt compiled from Langfuse",
                             mission_id=mission.get("id"))
                return lf_prompt, text
            except Exception as e:
                logger.warning(
                    "Langfuse compile failed for user_filter, using local fallback",
                    error=str(e),
                )

        # Fallback: local helper
        text = PromptTemplates.format_user_filter(
            signal=signal,
            mission=mission,
            existing_positions=existing_positions,
            margin_used=margin_used,
            account_value=account_value,
        )
        return lf_prompt, text

    def _build_user_filter_vars(
        self,
        signal: Dict[str, Any],
        mission: Dict[str, Any],
        existing_positions: List[Dict[str, Any]],
        margin_used: float,
        account_value: float,
    ) -> Dict[str, str]:
        """Build pre-formatted Mustache variables for user filter."""
        positions_str = "None" if not existing_positions else "\n".join([
            f"- {p['asset']}: {p['direction']} at {p['leverage']}x, PnL: {p['unrealized_pnl']:+.2f}%"
            for p in existing_positions
        ])

        open_position_count = len(existing_positions)

        # Actual margin utilization from clearinghouse
        if account_value > 0 and margin_used > 0:
            capital_deployed_percent = (margin_used / account_value) * 100
        elif open_position_count > 0:
            total_margin = sum(p.get("margin_used", 0) for p in existing_positions)
            current_val = mission.get("current_value", 0)
            if current_val > 0 and total_margin > 0:
                capital_deployed_percent = (total_margin / current_val) * 100
            else:
                capital_deployed_percent = open_position_count * 15.0
        else:
            capital_deployed_percent = 0.0

        # Correlation section
        from app.services.risk_manager import CORRELATION_BUCKETS
        asset_name = signal.get("asset", "")
        target_bucket = CORRELATION_BUCKETS.get(asset_name, "uncorrelated")
        same_bucket_positions = [
            p["asset"] for p in existing_positions
            if CORRELATION_BUCKETS.get(p["asset"], "uncorrelated") == target_bucket
        ]
        if same_bucket_positions:
            correlation_section = (
                f"- Correlation Group: {asset_name} is in '{target_bucket}' bucket. "
                f"Existing positions in same group: {', '.join(same_bucket_positions)}. "
                f"Adding another correlated position increases concentration risk."
            )
        else:
            correlation_section = (
                f"- Correlation Group: {asset_name} is in '{target_bucket}' bucket. "
                f"No existing positions in this group — diversification benefit."
            )

        current_capital = mission.get("current_value", 0)
        current_pnl = mission.get("total_pnl_percent", 0)
        max_leverage = mission.get("max_leverage", 2)

        return {
            "asset": signal.get("asset", ""),
            "direction": signal.get("direction", ""),
            "confidence": signal.get("confidence", ""),
            "recommended_leverage": str(signal.get("recommended_leverage", 1)),
            "strategy_tag": signal.get("strategy_tag", "unknown"),
            "risk_reward_ratio": str(signal.get("risk_reward_ratio", "N/A")),
            "funding_rate_impact": signal.get("funding_rate_impact", "neutral"),
            "signal_reasoning": signal.get("reasoning", ""),
            "risk_level": mission.get("risk_level", "MODERATE"),
            "max_leverage": str(max_leverage),
            "allowed_assets": ", ".join(mission.get("allowed_assets", [])),
            "mission_day": str(mission.get("mission_day", 1)),
            "mission_duration": str(mission.get("duration_days", 30)),
            "days_remaining": str(mission.get("days_remaining", 30)),
            "current_capital": f"{current_capital:,.2f}",
            "current_pnl": f"{current_pnl:+.2f}",
            "existing_positions": positions_str,
            "capital_deployed_percent": f"{capital_deployed_percent:.1f}",
            "open_position_count": str(open_position_count),
            "correlation_section": correlation_section,
        }

    # ── Position Exit Prompt ───────────────────────────────────────────

    def get_position_exit_prompt(
        self,
        position: Dict[str, Any],
        mission: Dict[str, Any],
        market_data: Dict[str, Any],
    ) -> Tuple[Optional[Any], str]:
        """
        Fetch and format the position exit prompt.

        Primary: Langfuse compile() with pre-formatted Mustache variables.
        Fallback: PromptTemplates.format_position_exit() local helper.
        """
        lf_prompt, _ = self._fetch(
            "position_exit", PromptTemplates.POSITION_EXIT_PROMPT
        )

        lf_variables = self._build_position_exit_vars(position, mission, market_data)

        if lf_prompt is not None:
            try:
                text = lf_prompt.compile(**lf_variables)
                logger.debug("Position exit prompt compiled from Langfuse",
                             asset=position.get("asset"))
                return lf_prompt, text
            except Exception as e:
                logger.warning(
                    "Langfuse compile failed for position_exit, using local fallback",
                    error=str(e),
                )

        # Fallback: local helper
        text = PromptTemplates.format_position_exit(
            position=position,
            mission=mission,
            market_data=market_data,
        )
        return lf_prompt, text

    def _build_position_exit_vars(
        self,
        position: Dict[str, Any],
        mission: Dict[str, Any],
        market_data: Dict[str, Any],
    ) -> Dict[str, str]:
        """Build pre-formatted Mustache variables for position exit."""
        raw_funding = market_data.get("funding_rate", 0)
        funding_rate_hourly = raw_funding * 100
        funding_rate_apr = raw_funding * 100 * 8760
        entry_price = position.get("entry_price", 0)
        current_price = position.get("current_price", 0)
        unrealized_pnl = position.get("unrealized_pnl_percent", 0)
        peak_pnl = position.get("peak_pnl_percent", 0)
        position_value = position.get("position_value", 0)
        total_pnl = mission.get("total_pnl_percent", 0)
        price_change_24h = market_data.get("price_change_24h", 0)
        current_volatility = market_data.get("volatility", 5)

        return {
            "asset": position.get("asset", ""),
            "direction": position.get("direction", ""),
            "leverage": str(position.get("leverage", 1)),
            "hours_in_position": str(position.get("hours_in_position", 0)),
            "mission_day": str(mission.get("mission_day", 1)),
            "mission_duration": str(mission.get("duration_days", 30)),
            "days_remaining": str(mission.get("days_remaining", 30)),
            "entry_price": f"{entry_price:,.2f}",
            "current_price": f"{current_price:,.2f}",
            "unrealized_pnl": f"{unrealized_pnl:+.2f}",
            "peak_pnl": f"{peak_pnl:+.2f}",
            "position_value": f"{position_value:,.2f}",
            "funding_rate_hourly": f"{funding_rate_hourly:.4f}",
            "funding_rate_apr": f"{funding_rate_apr:.1f}",
            "total_pnl": f"{total_pnl:+.2f}",
            "price_change_24h": f"{price_change_24h:+.2f}",
            "current_volatility": f"{current_volatility:.1f}",
        }


@lru_cache(maxsize=1)
def get_prompt_manager() -> PromptManager:
    """Get the singleton PromptManager."""
    return PromptManager()

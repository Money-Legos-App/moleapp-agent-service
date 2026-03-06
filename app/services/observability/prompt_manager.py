"""
Prompt Manager — fetches prompts from Langfuse with local fallback.

Usage:
    pm = get_prompt_manager()
    text, lf_prompt = pm.get_market_analysis_prompt(
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
    Fetches prompts from Langfuse (with in-SDK memory cache) and
    falls back to local PromptTemplates on failure.

    The lf_prompt object returned alongside the text must be passed
    to generation calls so Langfuse links them to specific prompt versions.
    """

    def __init__(self):
        self._lf = get_langfuse()

    def _fetch_raw(self, key: str, fallback_text: str) -> Tuple[Optional[Any], str]:
        """
        Fetch a raw (unformatted) prompt template.
        Returns (lf_prompt_obj, template_text).
        lf_prompt_obj is None when Langfuse is disabled or fetch fails.
        """
        lf_prompt, text = self._lf.get_prompt(
            prompt_name=PROMPT_NAMES[key],
            fallback=fallback_text,
        )
        if text is None:
            text = fallback_text
        return lf_prompt, text

    def get_system_prompt(self) -> Tuple[Optional[Any], str]:
        """Fetch the system prompt (no formatting needed)."""
        return self._fetch_raw("system", PromptTemplates.SYSTEM_PROMPT)

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
    ) -> Tuple[Optional[Any], str]:
        """
        Fetch and format the market analysis prompt.
        Returns (lf_prompt_obj, formatted_text).
        """
        lf_prompt, raw_template = self._fetch_raw(
            "market_analysis", PromptTemplates.MARKET_ANALYSIS_PROMPT
        )

        try:
            # Same formatting logic as PromptTemplates.format_market_analysis
            funding_rate_hourly = funding_rate * 100
            funding_rate_apr = funding_rate * 100 * 8760

            text = raw_template.format(
                asset=asset,
                current_price=current_price,
                price_change_24h=price_change_24h,
                volume_24h=volume_24h,
                spread=spread,
                funding_rate_hourly=funding_rate_hourly,
                funding_rate_apr=funding_rate_apr,
                open_interest=open_interest,
                pattern_context=pattern_context,
                max_drawdown_30d=risk_metrics.get("max_drawdown_30d", -20) * 100,
                volatility_30d=risk_metrics.get("volatility_30d", 5) * 100,
                sample_count=risk_metrics.get("sample_count", 0),
            )
            return lf_prompt, text
        except (KeyError, ValueError) as e:
            logger.warning("Langfuse prompt format error, using local template", error=str(e))
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
            )
            return None, text

    def get_user_filter_prompt(
        self,
        signal: Dict[str, Any],
        mission: Dict[str, Any],
        existing_positions: List[Dict[str, Any]],
    ) -> Tuple[Optional[Any], str]:
        """
        Fetch and format the user filter prompt.
        Returns (lf_prompt_obj, formatted_text).
        """
        lf_prompt, raw_template = self._fetch_raw(
            "user_filter", PromptTemplates.USER_FILTER_PROMPT
        )

        try:
            # Same formatting logic as PromptTemplates.format_user_filter
            positions_str = "None" if not existing_positions else "\n".join([
                f"- {p['asset']}: {p['direction']} at {p['leverage']}x, PnL: {p['unrealized_pnl']:+.2f}%"
                for p in existing_positions
            ])
            open_position_count = len(existing_positions)
            capital_deployed_percent = min(open_position_count * 15, 100)

            text = raw_template.format(
                asset=signal.get("asset"),
                direction=signal.get("direction"),
                confidence=signal.get("confidence"),
                recommended_leverage=signal.get("recommended_leverage", 1),
                strategy_tag=signal.get("strategy_tag", "unknown"),
                risk_reward_ratio=signal.get("risk_reward_ratio", "N/A"),
                funding_rate_impact=signal.get("funding_rate_impact", "neutral"),
                signal_reasoning=signal.get("reasoning", ""),
                risk_level=mission.get("risk_level", "MODERATE"),
                max_leverage=mission.get("max_leverage", 2),
                allowed_assets=", ".join(mission.get("allowed_assets", [])),
                mission_day=mission.get("mission_day", 1),
                mission_duration=mission.get("duration_days", 30),
                days_remaining=mission.get("days_remaining", 30),
                current_capital=mission.get("current_value", 0),
                current_pnl=mission.get("total_pnl_percent", 0),
                existing_positions=positions_str,
                capital_deployed_percent=capital_deployed_percent,
                open_position_count=open_position_count,
            )
            return lf_prompt, text
        except (KeyError, ValueError) as e:
            logger.warning("Langfuse prompt format error, using local template", error=str(e))
            text = PromptTemplates.format_user_filter(
                signal=signal,
                mission=mission,
                existing_positions=existing_positions,
            )
            return None, text

    def get_position_exit_prompt(
        self,
        position: Dict[str, Any],
        mission: Dict[str, Any],
        market_data: Dict[str, Any],
    ) -> Tuple[Optional[Any], str]:
        """
        Fetch and format the position exit prompt.
        Returns (lf_prompt_obj, formatted_text).
        """
        lf_prompt, raw_template = self._fetch_raw(
            "position_exit", PromptTemplates.POSITION_EXIT_PROMPT
        )

        try:
            # Same formatting logic as PromptTemplates.format_position_exit
            raw_funding = market_data.get("funding_rate", 0)
            funding_rate_hourly = raw_funding * 100
            funding_rate_apr = raw_funding * 100 * 8760

            text = raw_template.format(
                asset=position.get("asset"),
                direction=position.get("direction"),
                entry_price=position.get("entry_price", 0),
                current_price=position.get("current_price", 0),
                unrealized_pnl=position.get("unrealized_pnl_percent", 0),
                position_value=position.get("position_value", 0),
                leverage=position.get("leverage", 1),
                hours_in_position=position.get("hours_in_position", 0),
                funding_rate_hourly=funding_rate_hourly,
                funding_rate_apr=funding_rate_apr,
                mission_day=mission.get("mission_day", 1),
                mission_duration=mission.get("duration_days", 30),
                days_remaining=mission.get("days_remaining", 30),
                total_pnl=mission.get("total_pnl_percent", 0),
                price_change_24h=market_data.get("price_change_24h", 0),
                current_volatility=market_data.get("volatility", 5),
            )
            return lf_prompt, text
        except (KeyError, ValueError) as e:
            logger.warning("Langfuse prompt format error, using local template", error=str(e))
            text = PromptTemplates.format_position_exit(
                position=position,
                mission=mission,
                market_data=market_data,
            )
            return None, text


@lru_cache(maxsize=1)
def get_prompt_manager() -> PromptManager:
    """Get the singleton PromptManager."""
    return PromptManager()

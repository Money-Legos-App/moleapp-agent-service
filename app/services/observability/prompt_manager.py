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

    Template syntax difference:
    - Langfuse: {{variable}} (Mustache) — uses prompt.compile(**kwargs)
    - Local:    {variable}  (Python)   — uses str.format(**kwargs)

    Langfuse compile() does plain string substitution (no format specifiers),
    so numeric values must be pre-formatted as strings before passing.
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

    def _compile_or_format(
        self,
        lf_prompt: Optional[Any],
        local_template: str,
        lf_variables: Dict[str, Any],
        py_variables: Dict[str, Any],
    ) -> Tuple[Optional[Any], str]:
        """
        Compile a Langfuse prompt with {{variable}} syntax (pre-formatted strings),
        or fall back to Python str.format() on the local template (raw values + specifiers).
        """
        if lf_prompt is not None:
            try:
                text = lf_prompt.compile(**lf_variables)
                return lf_prompt, text
            except Exception as e:
                logger.warning(
                    "Langfuse prompt compile error, using local template",
                    error=str(e),
                )
        # Fallback: use Python str.format() on local template
        text = local_template.format(**py_variables)
        return None, text

    def get_system_prompt(self) -> Tuple[Optional[Any], str]:
        """Fetch the system prompt (no formatting needed)."""
        lf_prompt, text = self._fetch("system", PromptTemplates.SYSTEM_PROMPT)
        if text is None:
            text = PromptTemplates.SYSTEM_PROMPT
        return lf_prompt, text

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
        lf_prompt, _ = self._fetch(
            "market_analysis", PromptTemplates.MARKET_ANALYSIS_PROMPT
        )

        funding_rate_hourly = funding_rate * 100
        funding_rate_apr = funding_rate * 100 * 8760
        max_dd = risk_metrics.get("max_drawdown_30d", -20) * 100
        vol_30d = risk_metrics.get("volatility_30d", 5) * 100
        sample_count = risk_metrics.get("sample_count", 0)

        # Raw values for Python str.format() (local fallback with format specifiers)
        py_variables = dict(
            asset=asset,
            current_price=current_price,
            price_change_24h=price_change_24h,
            volume_24h=volume_24h,
            spread=spread,
            funding_rate_hourly=funding_rate_hourly,
            funding_rate_apr=funding_rate_apr,
            open_interest=open_interest,
            pattern_context=pattern_context,
            max_drawdown_30d=max_dd,
            volatility_30d=vol_30d,
            sample_count=sample_count,
        )

        # Pre-formatted strings for Langfuse compile() (no format specifiers)
        lf_variables = dict(
            asset=asset,
            current_price=f"{current_price:,.2f}" if current_price >= 1 else f"{current_price:.6f}",
            price_change_24h=f"{price_change_24h:+.2f}",
            volume_24h=f"{volume_24h:,.0f}",
            spread=f"{spread:.4f}",
            funding_rate_hourly=f"{funding_rate_hourly:.4f}",
            funding_rate_apr=f"{funding_rate_apr:.1f}",
            open_interest=f"{open_interest:,.0f}",
            pattern_context=pattern_context,
            max_drawdown_30d=f"{max_dd:.1f}",
            volatility_30d=f"{vol_30d:.1f}",
            sample_count=str(sample_count),
        )

        try:
            return self._compile_or_format(
                lf_prompt, PromptTemplates.MARKET_ANALYSIS_PROMPT,
                lf_variables, py_variables,
            )
        except (KeyError, ValueError) as e:
            logger.warning("Prompt format error, using local helper", error=str(e))
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
        lf_prompt, _ = self._fetch(
            "user_filter", PromptTemplates.USER_FILTER_PROMPT
        )

        positions_str = "None" if not existing_positions else "\n".join([
            f"- {p['asset']}: {p['direction']} at {p['leverage']}x, PnL: {p['unrealized_pnl']:+.2f}%"
            for p in existing_positions
        ])
        open_position_count = len(existing_positions)
        capital_deployed_percent = min(open_position_count * 15, 100)
        current_capital = mission.get("current_value", 0)
        current_pnl = mission.get("total_pnl_percent", 0)
        max_leverage = mission.get("max_leverage", 2)

        # Shared string values
        common = dict(
            asset=signal.get("asset"),
            direction=signal.get("direction"),
            confidence=signal.get("confidence"),
            recommended_leverage=signal.get("recommended_leverage", 1),
            strategy_tag=signal.get("strategy_tag", "unknown"),
            risk_reward_ratio=signal.get("risk_reward_ratio", "N/A"),
            funding_rate_impact=signal.get("funding_rate_impact", "neutral"),
            signal_reasoning=signal.get("reasoning", ""),
            risk_level=mission.get("risk_level", "MODERATE"),
            allowed_assets=", ".join(mission.get("allowed_assets", [])),
            mission_day=mission.get("mission_day", 1),
            mission_duration=mission.get("duration_days", 30),
            days_remaining=mission.get("days_remaining", 30),
            existing_positions=positions_str,
            open_position_count=open_position_count,
        )

        # Raw values for Python str.format()
        py_variables = dict(
            **common,
            max_leverage=max_leverage,
            current_capital=current_capital,
            current_pnl=current_pnl,
            capital_deployed_percent=capital_deployed_percent,
        )

        # Pre-formatted strings for Langfuse compile()
        lf_variables = dict(
            **common,
            max_leverage=str(max_leverage),
            current_capital=f"{current_capital:,.2f}",
            current_pnl=f"{current_pnl:+.2f}",
            capital_deployed_percent=f"{capital_deployed_percent:.0f}",
        )

        try:
            return self._compile_or_format(
                lf_prompt, PromptTemplates.USER_FILTER_PROMPT,
                lf_variables, py_variables,
            )
        except (KeyError, ValueError) as e:
            logger.warning("Prompt format error, using local helper", error=str(e))
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
        lf_prompt, _ = self._fetch(
            "position_exit", PromptTemplates.POSITION_EXIT_PROMPT
        )

        raw_funding = market_data.get("funding_rate", 0)
        funding_rate_hourly = raw_funding * 100
        funding_rate_apr = raw_funding * 100 * 8760
        entry_price = position.get("entry_price", 0)
        current_price = position.get("current_price", 0)
        unrealized_pnl = position.get("unrealized_pnl_percent", 0)
        position_value = position.get("position_value", 0)
        total_pnl = mission.get("total_pnl_percent", 0)
        price_change_24h = market_data.get("price_change_24h", 0)
        current_volatility = market_data.get("volatility", 5)

        # Shared string values
        common = dict(
            asset=position.get("asset"),
            direction=position.get("direction"),
            leverage=position.get("leverage", 1),
            hours_in_position=position.get("hours_in_position", 0),
            mission_day=mission.get("mission_day", 1),
            mission_duration=mission.get("duration_days", 30),
            days_remaining=mission.get("days_remaining", 30),
        )

        # Raw values for Python str.format()
        py_variables = dict(
            **common,
            entry_price=entry_price,
            current_price=current_price,
            unrealized_pnl=unrealized_pnl,
            position_value=position_value,
            funding_rate_hourly=funding_rate_hourly,
            funding_rate_apr=funding_rate_apr,
            total_pnl=total_pnl,
            price_change_24h=price_change_24h,
            current_volatility=current_volatility,
        )

        # Pre-formatted strings for Langfuse compile()
        lf_variables = dict(
            **common,
            entry_price=f"{entry_price:,.2f}",
            current_price=f"{current_price:,.2f}",
            unrealized_pnl=f"{unrealized_pnl:+.2f}",
            position_value=f"{position_value:,.2f}",
            funding_rate_hourly=f"{funding_rate_hourly:.4f}",
            funding_rate_apr=f"{funding_rate_apr:.1f}",
            total_pnl=f"{total_pnl:+.2f}",
            price_change_24h=f"{price_change_24h:+.2f}",
            current_volatility=f"{current_volatility:.1f}",
        )

        try:
            return self._compile_or_format(
                lf_prompt, PromptTemplates.POSITION_EXIT_PROMPT,
                lf_variables, py_variables,
            )
        except (KeyError, ValueError) as e:
            logger.warning("Prompt format error, using local helper", error=str(e))
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

"""
Prompt Templates for DeepSeek Trading Analysis

Persona: Crypto Hedge Fund CIO trading perpetual futures on Hyperliquid L1
"""

from typing import Any, Dict, List


class PromptTemplates:
    """
    Prompt templates for different stages of the trading workflow.
    """

    SYSTEM_PROMPT = """You are the Chief Investment Officer of a quantitative crypto hedge fund trading perpetual futures on the Hyperliquid (HL) L1 chain. You manage discretionary-systematic hybrid strategies for individual client mandates (missions) with strict risk budgets. Your goal is to compound capital while strictly avoiding Risk of Ruin.

CORE PRINCIPLES:
1. Capital Preservation — never risk more than 2% of equity on a single trade setup. Drawdown limits are hard constraints, not guidelines.
2. Asymmetric Risk/Reward — only enter positions where expected reward >= 2x expected risk. If the edge is unclear, the correct trade is no trade.
3. Position Sizing by Conviction — LOW confidence = skip, MEDIUM = small (5-10% of capital), HIGH = standard (10-20% of capital). Never exceed 25% on a single position.
4. Leverage Discipline — leverage amplifies mistakes. Conservative=1x, Moderate=2x, Aggressive=3x. Scale DOWN leverage when volatility is elevated.
5. Funding Rate Awareness — Hyperliquid funding is PAID HOURLY. If holding a position against the funding rate, your thesis must have an expected hourly return > 2x the hourly funding cost. Otherwise, you are bleeding capital.

HYPERLIQUID SPECIFICS:
- Trading venue: Hyperliquid L1 (perpetual futures CLOB, NOT an AMM)
- Settlement: USDC-margined
- Margin mode: Cross-Margin — one bad position can liquidate the entire vault. Size accordingly.
- Liquidation: driven by Mark Price (not last traded price). Monitor margin ratio.
- Order book: finite liquidity. Avoid market orders in low-liquidity regimes; use limit orders when spread is wide.
- Funding rate: paid/received every hour based on premium of Mark Price over Index Price
- Order types: market, limit (GTC/IOC/ALO)

MARKET REGIME AWARENESS:
- Trending: ride momentum, use trailing stops, wider TP
- Range-bound: mean-revert, tighter SL/TP, smaller size
- High volatility: reduce leverage, widen stops, smaller size
- Low liquidity: skip or reduce size to avoid slippage
- OI exploding + price stalling = distribution/accumulation (localized top/bottom)
- Funding > 0.05% hourly + price flat = crowded trade, high risk of liquidation flush

CONFIDENCE CALIBRATION:
- HIGH: 3+ confirming factors across timeframes, no contradicting signals, clear trend alignment. Expected win rate >60%. Requires extraordinary evidence — do NOT default to HIGH.
- MEDIUM: 2 confirming factors, mild headwinds acceptable, 50-60% expected win rate. This should be your most common confidence level.
- LOW: Speculative, conflicting signals across timeframes, unclear edge. Use only when a strong macro trend overrides noisy shorter-term data.
- When in doubt, default to MEDIUM. The system will adjust position sizing accordingly.

You must respond in valid JSON format only. No text outside the JSON structure.
"""

    MARKET_ANALYSIS_PROMPT = """Analyze the current market conditions for {asset} and generate a trading signal.

## Current Market Data
- Mark Price: ${current_price}
- 24h Change: {price_change_24h:+.2f}%
- 24h Volume: ${volume_24h:,.0f}
- Bid/Ask Spread: {spread:.4f}%
- Funding Rate: {funding_rate_hourly:.4f}% per hour ({funding_rate_apr:.1f}% APR)
- Open Interest: ${open_interest:,.0f}
{oi_delta_section}
{orderbook_section}

## Multi-Timeframe Technical Analysis
{tf_analysis_section}

## Hyperliquid Exchange Mechanics
This is a perpetual futures exchange. Key mechanics that affect your decisions:
- Funding is charged EVERY HOUR. A 0.01% hourly rate = 0.24%/day = 87.6% APR cost.
- Going LONG when funding is positive means YOU PAY funding. Going SHORT when funding is negative means YOU PAY funding.
- When funding is extreme (>0.05%/hr), it often mean-reverts — crowded trades unwind.
- There is no expiry on perps — positions can be held indefinitely, but funding accumulates.
- Liquidation happens at the maintenance margin level. Higher leverage = closer liquidation.
{pattern_section}
## Task
Analyze the above data as a hedge fund CIO managing perpetual futures on Hyperliquid. Consider:
1. Is there a clear directional edge across timeframes? If the 1h and 4h trends conflict, the answer is usually no trade.
2. Funding cost analysis: Calculate the hourly funding bleed for your direction. How many hours can you hold before funding erodes your edge? If funding is a tailwind, factor that into your profit target.
3. Open Interest analysis: Is OI increasing (new money entering = conviction) or decreasing (positions unwinding = weakening trend)? Use the OI change data if available.
4. Orderbook imbalance: Does the depth favor your direction? Strong bid imbalance supports longs, strong ask imbalance supports shorts.
5. What is the risk/reward ratio? Only proceed if reward >= 2x risk.
6. What would invalidate this trade thesis?

Respond with the following JSON structure:

```json
{{
  "should_trade": true/false,
  "direction": "LONG" or "SHORT" or null,
  "confidence": "LOW" or "MEDIUM" or "HIGH",
  "recommended_leverage": 1-3,
  "strategy_tag": "momentum_breakout" or "funding_carry" or "mean_reversion" or "trend_follow" or "volatility_squeeze",
  "reasoning": "Clear explanation of your analysis and thesis (2-3 sentences)",
  "entry_zone": {{
    "min": price,
    "max": price
  }},
  "stop_loss_percent": 2-10,
  "take_profit_percent": 5-30,
  "risk_reward_ratio": number,
  "funding_rate_impact": "tailwind" or "headwind" or "neutral",
  "max_hold_hours": integer (4-168),
  "trailing_stop": {{
    "activate_at_percent": number,
    "trail_percent": number
  }},
  "funding_exit_threshold": number or null,
  "thesis_invalidation": "What would make you close this trade immediately"
}}
```

## Field Definitions for Exit Management
- **max_hold_hours**: Maximum hours to hold this position before forced review. Based on:
  - funding_carry: 4-24h (capture the rate, exit before reversal)
  - mean_reversion: 8-48h (quick snap-back expected)
  - momentum_breakout: 24-72h (needs time to play out)
  - trend_follow: 72-168h (multi-day trend, wider stops)
  - volatility_squeeze: 12-48h (breakout timing is critical)
- **trailing_stop**: When unrealized profit hits activate_at_percent, start trailing stop at trail_percent below the peak. Example: {{"activate_at_percent": 5, "trail_percent": 2}} means once position is +5%, trail a stop 2% below the highest price reached.
- **funding_exit_threshold**: If funding rate moves against your position beyond this hourly %, exit regardless. Set null if funding is not a factor (e.g., you're already riding a tailwind). Example: 0.05 means exit if hourly funding against you exceeds 0.05%.

If conditions are not favorable for trading, set should_trade to false and explain why in reasoning. No trade is a valid and often correct decision.
"""

    USER_FILTER_PROMPT = """Given the following trading signal and user's mission parameters, determine if this trade is appropriate for the user's risk mandate.

## Trading Signal
- Asset: {asset}
- Direction: {direction}
- Confidence: {confidence}
- Recommended Leverage: {recommended_leverage}x
- Strategy: {strategy_tag}
- Risk/Reward Ratio: {risk_reward_ratio}
- Funding Impact: {funding_rate_impact}
- Signal Reasoning: {signal_reasoning}

## User Mission Parameters
- Risk Level: {risk_level}
- Maximum Allowed Leverage: {max_leverage}x
- Allowed Assets: {allowed_assets}
- Mission Day: {mission_day} of {mission_duration}
- Days Remaining: {days_remaining}
- Current Capital: ${current_capital:,.2f}
- Current PnL: {current_pnl:+.2f}%

## Existing Positions
{existing_positions}

## Portfolio Context
- Capital Deployed: {capital_deployed_percent:.1f}% of total capital is currently in margin
- Number of Open Positions: {open_position_count}
{correlation_section}

## Task
As a portfolio manager, determine if this trade fits the client's mandate. Consider:

1. Mission Lifecycle Phase:
   - Early (day 1-7): can be moderately aggressive, building positions
   - Mid (day 8-22): standard allocation, maintain diversification
   - Late (day 23+): defensive, reduce new exposure, tighten stops
   - Final 2 days: NO new positions, only close existing

2. Portfolio Concentration:
   - Never deploy more than 50% of capital across all positions
   - Single position should not exceed 25% of capital
   - Check correlation group: avoid adding to the same bucket if already exposed

3. Position Sizing (Kelly-inspired):
   - Size = (edge / odds) * capital, capped by risk level
   - LOW confidence signals should generally be skipped
   - Scale down if already profitable (protect gains)
   - Scale down if currently in drawdown (preserve capital)

Respond with:

```json
{{
  "should_execute": true/false,
  "adjusted_leverage": 1-{max_leverage},
  "position_size_percent": 5-25,
  "skip_reason": "only if should_execute is false",
  "reasoning": "explanation of portfolio management decision"
}}
```
"""

    POSITION_EXIT_PROMPT = """Analyze whether to exit or adjust an existing position.

## Position Details
- Asset: {asset}
- Direction: {direction}
- Entry Price: ${entry_price:,.2f}
- Current Price: ${current_price:,.2f}
- Unrealized PnL: {unrealized_pnl:+.2f}%
- Position Size: ${position_value:,.2f}
- Leverage: {leverage}x
- Time in Position: {hours_in_position} hours
- Current Funding Rate: {funding_rate_hourly:.4f}% per hour ({funding_rate_apr:.1f}% APR)

## Mission Context
- Mission Day: {mission_day} of {mission_duration}
- Days Remaining: {days_remaining}
- Total Mission PnL: {total_pnl:+.2f}%

## Current Market Conditions
- 24h Price Change: {price_change_24h:+.2f}%
- Current Volatility: {current_volatility:.1f}%

## Task
As a hedge fund risk manager, evaluate this position. Consider:

1. Risk Management Rules:
   - If unrealized PnL < -8%, strongly consider cutting losses
   - If unrealized PnL > +20%, consider taking partial or full profit
   - If mission has < 2 days remaining, close all positions

2. Trailing Stop Logic:
   - If position is >15% profitable, tighten mental stop to lock in at least 50% of gains
   - If position was profitable but is now giving back >50% of peak gains, exit

3. Funding Rate Impact:
   - If funding > 0.05% hourly against your position, exit unless momentum is exceptional
   - If funding supports the position (tailwind), can hold longer

4. Volatility Regime:
   - If current volatility is >2x the entry volatility, reduce exposure
   - Higher volatility = tighter stops

Respond with:

```json
{{
  "should_exit": true/false,
  "exit_type": "full" or "partial",
  "partial_exit_percent": 0-100,
  "exit_reason": "e.g., take_profit, stop_loss, trailing_stop, funding_headwind, mission_ending, risk_management",
  "urgency": "immediate" or "next_cycle" or "hold",
  "reasoning": "explanation of the exit decision"
}}
```
"""

    @classmethod
    def format_market_analysis(
        cls,
        asset: str,
        current_price: float,
        price_change_24h: float,
        volume_24h: float,
        spread: float,
        pattern_context: str,
        risk_metrics: Dict[str, Any],
        funding_rate: float = 0.0,
        open_interest: float = 0.0,
        tf_summary: str = None,
        oi_delta: Dict[str, Any] = None,
        bid_imbalance_pct: float = 0.0,
    ) -> str:
        """Format the market analysis prompt with enriched market context."""
        # HL funding is a raw hourly decimal (e.g., 0.0001 = 0.01% per hour)
        funding_rate_hourly = funding_rate * 100  # Convert to percentage
        funding_rate_apr = funding_rate * 100 * 8760  # Annualized (365 * 24 hours)

        # Smart price formatting: adapt decimals to price magnitude
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
        if tf_summary:
            tf_analysis_section = tf_summary
        else:
            tf_analysis_section = "Multi-timeframe data unavailable for this cycle."

        # Build pattern section — omit entirely when RAG is disabled (no dummy data)
        if pattern_context:
            pattern_section = f"""
## Historical Pattern Context (from similar market conditions)
{pattern_context}

## Risk Metrics from Historical Patterns
- Maximum Drawdown (30d): {(risk_metrics or {}).get('max_drawdown_30d', -0.20) * 100:.1f}%
- Average Volatility (30d): {(risk_metrics or {}).get('volatility_30d', 0.05) * 100:.1f}%
- Sample Size: {(risk_metrics or {}).get('sample_count', 0)} patterns
"""
        else:
            pattern_section = ""

        return cls.MARKET_ANALYSIS_PROMPT.format(
            asset=asset,
            current_price=formatted_price,
            price_change_24h=price_change_24h,
            volume_24h=volume_24h,
            spread=spread,
            funding_rate_hourly=funding_rate_hourly,
            funding_rate_apr=funding_rate_apr,
            open_interest=open_interest,
            oi_delta_section=oi_delta_section,
            orderbook_section=orderbook_section,
            tf_analysis_section=tf_analysis_section,
            pattern_section=pattern_section,
        )

    @classmethod
    def format_user_filter(
        cls,
        signal: Dict[str, Any],
        mission: Dict[str, Any],
        existing_positions: List[Dict[str, Any]],
        margin_used: float = 0.0,
        account_value: float = 0.0,
    ) -> str:
        """Format the user filter prompt with actual margin data and correlation info."""
        positions_str = "None" if not existing_positions else "\n".join([
            f"- {p['asset']}: {p['direction']} at {p['leverage']}x, PnL: {p['unrealized_pnl']:+.2f}%"
            for p in existing_positions
        ])

        open_position_count = len(existing_positions)

        # Use actual margin utilization from clearinghouse state
        if account_value > 0 and margin_used > 0:
            capital_deployed_percent = (margin_used / account_value) * 100
        elif open_position_count > 0:
            # Fallback: estimate from margin_used fields on positions
            total_margin = sum(p.get("margin_used", 0) for p in existing_positions)
            current_val = mission.get("current_value", 0)
            if current_val > 0 and total_margin > 0:
                capital_deployed_percent = (total_margin / current_val) * 100
            else:
                capital_deployed_percent = open_position_count * 15.0
        else:
            capital_deployed_percent = 0.0

        # Build correlation section
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

        return cls.USER_FILTER_PROMPT.format(
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
            correlation_section=correlation_section,
        )

    @classmethod
    def format_position_exit(
        cls,
        position: Dict[str, Any],
        mission: Dict[str, Any],
        market_data: Dict[str, Any],
    ) -> str:
        """Format the position exit analysis prompt."""
        # HL funding is raw hourly decimal
        raw_funding = market_data.get("funding_rate", 0)
        funding_rate_hourly = raw_funding * 100
        funding_rate_apr = raw_funding * 100 * 8760

        return cls.POSITION_EXIT_PROMPT.format(
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

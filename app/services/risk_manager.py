"""
Ruthless Risk Manager

Deterministic, math-based position risk enforcement.
No LLM calls — pure math that runs every 5 minutes.

Risk thresholds are derived from the user's chosen riskLevel
(CONSERVATIVE / MODERATE / AGGRESSIVE) at mission creation.
"""

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import structlog

from app.config import get_settings

logger = structlog.get_logger(__name__)

# ==================
# Strategy-Aware Risk Profiles
# ==================

RISK_PROFILES = {
    "CONSERVATIVE": {
        "stop_loss_pct": 3.0,
        "take_profit_pct": 6.0,
        "trailing_activation_pct": 1.5,
        "trailing_callback_pct": 1.0,
        "max_drawdown_pct": 10.0,
        "max_funding_cost_pct": 1.0,
        "max_leverage": 1,
        "max_positions": 2,
        "max_per_bucket": 1,           # must diversify across buckets
        "max_margin_utilization": 0.25, # halt new entries above 25%
    },
    "MODERATE": {
        "stop_loss_pct": 5.0,
        "take_profit_pct": 10.0,
        "trailing_activation_pct": 2.0,
        "trailing_callback_pct": 1.5,
        "max_drawdown_pct": 20.0,
        "max_funding_cost_pct": 2.0,
        "max_leverage": 2,
        "max_positions": 4,
        "max_per_bucket": 2,           # max 2 per correlation sector
        "max_margin_utilization": 0.50, # halt new entries above 50%
    },
    "AGGRESSIVE": {
        "stop_loss_pct": 8.0,
        "take_profit_pct": 20.0,
        "trailing_activation_pct": 3.0,
        "trailing_callback_pct": 2.5,
        "max_drawdown_pct": 30.0,
        "max_funding_cost_pct": 3.0,
        "max_leverage": 3,
        "max_positions": 5,
        "max_per_bucket": 3,           # bounded by bucket caps
        "max_margin_utilization": 0.75, # halt new entries above 75%
    },
}


def get_risk_profile(risk_level: str) -> dict:
    """Get the risk profile for a given risk level."""
    return RISK_PROFILES.get(risk_level, RISK_PROFILES["MODERATE"])


# ==================
# Correlation Bucketing
# ==================

# Maps each tradeable asset to its correlation bucket
CORRELATION_BUCKETS = {
    "BTC-USD": "btc_correlated",
    "DOGE-USD": "btc_correlated",
    "ETH-USD": "eth_correlated",
    "ARB-USD": "eth_correlated",
    "SOL-USD": "sol_ecosystem",
    "WIF-USD": "sol_ecosystem",
    "POPCAT-USD": "sol_ecosystem",
    "SUI-USD": "alt_l1",
    "SEI-USD": "alt_l1",
    "AVAX-USD": "alt_l1",
    "NEAR-USD": "alt_l1",
    "TAO-USD": "ai_narrative",
    "FET-USD": "ai_narrative",
    "PENDLE-USD": "defi",
    "kPEPE-USD": "meme",
}

# Default caps per sector (overridden by settings at runtime)
BUCKET_CAPS = {
    "btc_correlated": 40.0,
    "eth_correlated": 30.0,
    "sol_ecosystem": 25.0,
    "alt_l1": 20.0,
    "ai_narrative": 20.0,
    "defi": 15.0,
    "meme": 15.0,
    "uncorrelated": 15.0,  # fallback for dynamically rotated assets
}


def check_correlation_bucket_exceeded(
    new_asset: str,
    new_leverage: float,
    new_margin: float,
    existing_positions: List[Dict[str, Any]],
    account_value: float,
    max_mission_leverage: float,
    bucket_caps: Optional[Dict[str, float]] = None,
) -> Tuple[bool, str]:
    """
    Check if adding a new position would exceed the bucket's leverage cap.

    Args:
        new_asset: Asset to add (e.g., "ETH-USD")
        new_leverage: Leverage for the new position
        new_margin: Margin allocated to the new position
        existing_positions: List of dicts with {asset, leverage, margin_used}
        account_value: Total account equity
        max_mission_leverage: Mission's max leverage setting
        bucket_caps: Override caps per bucket (default: BUCKET_CAPS)

    Returns:
        (exceeded, reason_string)
    """
    if account_value <= 0:
        return False, ""

    caps = bucket_caps or BUCKET_CAPS
    target_bucket = CORRELATION_BUCKETS.get(new_asset, "uncorrelated")
    cap_pct = caps.get(target_bucket, 100.0)

    # Total leverage budget = account_value * max_mission_leverage
    total_budget = account_value * max_mission_leverage
    if total_budget <= 0:
        return False, ""

    # Sum notional exposure in the target bucket from existing positions
    bucket_exposure = 0.0
    for pos in existing_positions:
        pos_bucket = CORRELATION_BUCKETS.get(pos.get("asset", ""), "uncorrelated")
        if pos_bucket == target_bucket:
            pos_leverage = pos.get("leverage", 1)
            pos_margin = pos.get("margin_used", 0)
            bucket_exposure += pos_leverage * pos_margin

    # Add proposed position
    proposed_total = bucket_exposure + (new_leverage * new_margin)
    bucket_pct = (proposed_total / total_budget) * 100

    if bucket_pct > cap_pct:
        return True, f"bucket_{target_bucket}_exceeded"

    return False, ""


# ==================
# Dynamic Stop-Loss Scaling
# ==================

# Approximate maintenance margin ratios for Hyperliquid assets
MAINTENANCE_MARGIN_RATIOS = {
    "BTC-USD": 0.0333,
    "ETH-USD": 0.0333,
    "SOL-USD": 0.05,
    "ARB-USD": 0.05,
    "OP-USD": 0.05,
    "XRP-USD": 0.05,
    "ADA-USD": 0.05,
    "DOGE-USD": 0.05,
    "LINK-USD": 0.05,
    "AVAX-USD": 0.05,
    "SUI-USD": 0.05,
    "WIF-USD": 0.10,
    "kPEPE-USD": 0.10,
}
DEFAULT_MAINTENANCE_MARGIN = 0.05


def calculate_dynamic_stop_loss_pct(
    asset: str,
    leverage: float,
    profile_sl_pct: float,
    liquidation_buffer_pct: float = 20.0,
) -> float:
    """
    Calculate leverage-aware stop loss percentage.

    Key distinction: 1/leverage is the distance to BANKRUPTCY (margin = $0),
    but Hyperliquid liquidates at the Maintenance Margin Requirement (MMR)
    BEFORE bankruptcy. The actual liquidation distance is (1/leverage - MMR).

    Without subtracting MMR, at high leverage the SL and liquidation trigger
    can race — if the exchange wins, the user pays a liquidation penalty.

    Logic:
    - Calculate true liquidation distance: (1/leverage) - MMR
    - Apply safety buffer (20% of that distance)
    - Use min(profile_sl, dynamic_sl) so SL always triggers before liquidation
    - At low leverage (1-3x), profile SL wins (dynamic is huge)
    - At high leverage (10x+), dynamic SL forces tighter stops

    Args:
        asset: Trading pair (e.g., "ETH-USD")
        leverage: Actual leverage used
        profile_sl_pct: Base SL% from risk profile (e.g., 5.0)
        liquidation_buffer_pct: Safety buffer as % of liquidation distance (default 20%)

    Returns:
        SL percentage to use
    """
    if leverage <= 0:
        return profile_sl_pct

    mmr = MAINTENANCE_MARGIN_RATIOS.get(asset, DEFAULT_MAINTENANCE_MARGIN)

    # Distance to LIQUIDATION (not bankruptcy):
    # bankruptcy_distance = 1/leverage
    # liquidation_distance = bankruptcy_distance - MMR
    # e.g., 20x BTC: (1/20) - 0.0333 = 0.0167 (1.67%)
    # e.g., 2x ETH:  (1/2)  - 0.0333 = 0.4667 (46.67%)
    liquidation_distance = (1.0 / leverage) - mmr

    # Floor at 0.5% to handle extreme leverage where liq_distance goes negative
    liquidation_distance = max(0.005, liquidation_distance)

    # Apply safety buffer: SL triggers at (1 - buffer%) of the liquidation distance
    # e.g., 20x BTC: 0.0167 * 0.80 = 0.01336 → SL at 1.34%, liq at 1.67%, gap = 0.33%
    dynamic_sl_pct = liquidation_distance * (1.0 - liquidation_buffer_pct / 100) * 100

    if dynamic_sl_pct <= 0:
        return 0.5

    # Use the STRICTER (smaller) of profile SL and dynamic SL
    return min(profile_sl_pct, dynamic_sl_pct)


# ==================
# Slippage-Aware Position Sizing
# ==================

def calculate_slippage_adjusted_size(
    base_position_size: float,
    market_price: float,
    cached_price_data: Dict[str, Any],
    max_slippage_reduction_pct: float = 30.0,
) -> Tuple[float, float]:
    """
    Reduce position size based on market stress indicators.

    Stress signals (from Redis-cached market data):
    - High absolute funding rate (crowded trade, potential cascade)
    - Mark price deviation from expected price (fast-moving market)

    Args:
        base_position_size: Original calculated position size
        market_price: Current market price used for the order
        cached_price_data: Redis cached data {markPx, funding, volume, OI}
        max_slippage_reduction_pct: Maximum reduction (default 30%)

    Returns:
        (adjusted_size, reduction_pct)
    """
    if not cached_price_data or base_position_size <= 0:
        return base_position_size, 0.0

    stress_score = 0.0  # 0.0 to 1.0

    # Factor 1: Funding rate stress (crowded trade indicator)
    try:
        funding = abs(float(cached_price_data.get("funding", 0)))
    except (ValueError, TypeError):
        funding = 0.0

    if funding > 0.0003:       # > 0.03% per 8h ≈ very stressed
        stress_score += 0.4
    elif funding > 0.0001:     # > 0.01% per 8h ≈ moderately stressed
        stress_score += 0.2

    # Factor 2: Mark price deviation from our order price
    try:
        mark_px = float(cached_price_data.get("markPx", 0))
    except (ValueError, TypeError):
        mark_px = 0.0

    if mark_px > 0 and market_price > 0:
        deviation_pct = abs(market_price - mark_px) / mark_px * 100
        if deviation_pct > 0.5:
            stress_score += 0.3
        elif deviation_pct > 0.2:
            stress_score += 0.15

    stress_score = min(stress_score, 1.0)

    reduction_pct = stress_score * max_slippage_reduction_pct
    adjusted_size = base_position_size * (1 - reduction_pct / 100)

    return max(adjusted_size, 0.0), round(reduction_pct, 2)


# ==================
# Risk Action
# ==================

@dataclass
class RiskAction:
    """A forced close action triggered by the risk manager."""
    position_id: str
    mission_id: str
    asset: str
    direction: str
    size: float
    reason: str          # STOP_LOSS, TAKE_PROFIT, TRAILING_STOP, LIQUIDATION_PROTECTION, FUNDING_EXIT, DRAWDOWN_KILL
    current_price: float
    trigger_price: float  # The threshold that was breached


# ==================
# Pure Check Functions (stateless, no I/O)
# ==================

def check_stop_loss(direction: str, current_price: float, sl_price: float) -> bool:
    """Check if stop loss has been hit."""
    if sl_price <= 0:
        return False
    if direction == "LONG":
        return current_price <= sl_price
    else:
        return current_price >= sl_price


def check_take_profit(direction: str, current_price: float, tp_price: float) -> bool:
    """Check if take profit has been hit."""
    if tp_price <= 0:
        return False
    if direction == "LONG":
        return current_price >= tp_price
    else:
        return current_price <= tp_price


def check_trailing_stop(
    direction: str,
    current_price: float,
    entry_price: float,
    peak_price: float,
    activation_pct: float,
    callback_pct: float,
) -> Tuple[bool, Optional[float]]:
    """
    Check trailing stop.

    Returns:
        (triggered, new_trailing_stop_price)
        - triggered: True if the trailing stop has been hit
        - new_trailing_stop_price: Updated trailing stop price (None if not yet activated)
    """
    if entry_price <= 0:
        return False, None

    if direction == "LONG":
        profit_pct = ((peak_price - entry_price) / entry_price) * 100
        if profit_pct < activation_pct:
            return False, None
        trailing_stop = peak_price * (1 - callback_pct / 100)
        return current_price <= trailing_stop, trailing_stop
    else:
        profit_pct = ((entry_price - peak_price) / entry_price) * 100
        if profit_pct < activation_pct:
            return False, None
        trailing_stop = peak_price * (1 + callback_pct / 100)
        return current_price >= trailing_stop, trailing_stop


def check_liquidation_proximity(
    direction: str,
    current_price: float,
    liquidation_price: float,
    buffer_pct: float,
) -> bool:
    """Check if price is within buffer_pct of liquidation."""
    if liquidation_price <= 0:
        return False
    if direction == "LONG":
        distance_pct = ((current_price - liquidation_price) / current_price) * 100
    else:
        distance_pct = ((liquidation_price - current_price) / current_price) * 100
    return distance_pct < buffer_pct


def check_funding_exit(
    funding_paid: float,
    margin_used: float,
    max_funding_cost_pct: float,
) -> bool:
    """Check if cumulative funding cost exceeds threshold."""
    if margin_used <= 0:
        return False
    cost_pct = (abs(funding_paid) / margin_used) * 100
    return cost_pct > max_funding_cost_pct


def check_mission_drawdown(
    current_value: float,
    initial_capital: float,
    max_drawdown_pct: float,
) -> bool:
    """Check if mission-level drawdown exceeds kill switch threshold."""
    if initial_capital <= 0:
        return False
    drawdown_pct = ((initial_capital - current_value) / initial_capital) * 100
    return drawdown_pct > max_drawdown_pct


# ==================
# Trailing Stop State (Redis)
# ==================

TRAILING_KEY_PREFIX = "agent:risk:trailing"
TRAILING_TTL = 7 * 24 * 3600  # 7 days


async def get_trailing_state(redis, position_id: str) -> Optional[Dict[str, float]]:
    """Get trailing stop state from Redis."""
    key = f"{TRAILING_KEY_PREFIX}:{position_id}"
    data = await redis.hgetall(key)
    if not data:
        return None
    return {
        "highest_price": float(data.get("highest_price", 0)),
        "lowest_price": float(data.get("lowest_price", 0)),
        "peak_pnl_percent": float(data.get("peak_pnl_percent", 0)),
    }


async def set_trailing_state(
    redis,
    position_id: str,
    highest_price: float,
    lowest_price: float,
    peak_pnl_percent: float = 0.0,
) -> None:
    """Set trailing stop state in Redis (includes peak PnL for exit prompt)."""
    key = f"{TRAILING_KEY_PREFIX}:{position_id}"
    pipe = redis.pipeline()
    pipe.hset(key, mapping={
        "highest_price": str(highest_price),
        "lowest_price": str(lowest_price),
        "peak_pnl_percent": str(peak_pnl_percent),
    })
    pipe.expire(key, TRAILING_TTL)
    await pipe.execute()


async def delete_trailing_state(redis, position_id: str) -> None:
    """Delete trailing stop state from Redis."""
    key = f"{TRAILING_KEY_PREFIX}:{position_id}"
    await redis.delete(key)


# ==================
# Orchestrator
# ==================

async def evaluate_mission_risk(
    mission: Dict[str, Any],
    hl_positions: List[Dict[str, Any]],
    account_value: float,
    db_positions: List[Dict[str, Any]],
    cached_prices: Dict[str, Any],
    redis,
) -> Tuple[List[RiskAction], bool]:
    """
    Evaluate all risk checks for a mission.

    Args:
        mission: Mission dict from get_active_missions()
        hl_positions: Positions from HL clearinghouse state
        account_value: Current account value from HL
        db_positions: Open positions from our DB (with SL/TP)
        cached_prices: Redis-cached mark prices
        redis: Redis connection for trailing stop state

    Returns:
        (actions, kill_switch)
        - actions: List of RiskAction to execute
        - kill_switch: True if ALL positions should be closed (mission drawdown)
    """
    settings = get_settings()
    profile = get_risk_profile(mission.get("risk_level", "MODERATE"))
    liq_buffer_pct = settings.risk_liquidation_protection_percent

    actions: List[RiskAction] = []
    mission_id = mission["id"]
    initial_capital = mission.get("initial_capital", 0)

    # Build lookup: asset -> DB position (for SL/TP prices)
    db_pos_by_asset = {}
    for dbp in db_positions:
        db_pos_by_asset[dbp["asset"]] = dbp

    for hl_pos in hl_positions:
        asset = hl_pos["asset"]
        direction = hl_pos["direction"]
        coin = asset.replace("-USD", "")

        # Resolve current price: cached mark price > HL position entry price
        if coin in cached_prices:
            current_price = cached_prices[coin].get("markPx", hl_pos["entry_price"])
        else:
            current_price = hl_pos["entry_price"]

        if current_price <= 0:
            continue

        size = abs(hl_pos.get("size", 0))
        if size <= 0:
            continue

        # Match to DB position for SL/TP
        db_pos = db_pos_by_asset.get(asset, {})
        position_id = db_pos.get("id", f"hl_{asset}_{mission_id}")
        entry_price = db_pos.get("entry_price", hl_pos.get("entry_price", 0))
        sl_price = db_pos.get("stop_loss_price", 0) or 0
        tp_price = db_pos.get("take_profit_price", 0) or 0
        liq_price = hl_pos.get("liquidation_price", 0) or 0
        funding_paid = db_pos.get("funding_paid", 0)
        margin_used = db_pos.get("margin_used", 0) or hl_pos.get("margin_used", 0)

        # Priority 1: Liquidation protection
        if liq_price > 0 and check_liquidation_proximity(direction, current_price, liq_price, liq_buffer_pct):
            actions.append(RiskAction(
                position_id=position_id,
                mission_id=mission_id,
                asset=asset,
                direction=direction,
                size=size,
                reason="LIQUIDATION_PROTECTION",
                current_price=current_price,
                trigger_price=liq_price,
            ))
            logger.error(
                "RISK: Liquidation protection triggered",
                mission_id=mission_id, asset=asset, direction=direction,
                current_price=current_price, liq_price=liq_price,
            )
            continue  # Skip other checks — this position is being closed

        # Priority 2: Stop loss
        if sl_price > 0 and check_stop_loss(direction, current_price, sl_price):
            actions.append(RiskAction(
                position_id=position_id,
                mission_id=mission_id,
                asset=asset,
                direction=direction,
                size=size,
                reason="STOP_LOSS",
                current_price=current_price,
                trigger_price=sl_price,
            ))
            logger.warning(
                "RISK: Stop loss triggered",
                mission_id=mission_id, asset=asset, direction=direction,
                current_price=current_price, sl_price=sl_price,
                risk_level=mission.get("risk_level"),
            )
            continue

        # Priority 3: Trailing stop
        trailing_state = await get_trailing_state(redis, position_id)
        if trailing_state:
            peak = trailing_state["highest_price"] if direction == "LONG" else trailing_state["lowest_price"]
        else:
            peak = current_price

        # Update peak price and peak PnL
        if direction == "LONG":
            new_highest = max(peak, current_price)
            new_lowest = trailing_state["lowest_price"] if trailing_state else current_price
        else:
            new_highest = trailing_state["highest_price"] if trailing_state else current_price
            new_lowest = min(peak, current_price)

        # Track peak unrealized PnL % (used by LLM exit prompt)
        if entry_price > 0:
            if direction == "LONG":
                current_pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:
                current_pnl_pct = ((entry_price - current_price) / entry_price) * 100
        else:
            current_pnl_pct = 0
        prev_peak_pnl = trailing_state.get("peak_pnl_percent", 0) if trailing_state else 0
        new_peak_pnl = max(prev_peak_pnl, current_pnl_pct)

        await set_trailing_state(redis, position_id, new_highest, new_lowest, new_peak_pnl)

        triggered, trailing_stop_price = check_trailing_stop(
            direction=direction,
            current_price=current_price,
            entry_price=entry_price,
            peak_price=new_highest if direction == "LONG" else new_lowest,
            activation_pct=profile["trailing_activation_pct"],
            callback_pct=profile["trailing_callback_pct"],
        )

        if triggered:
            actions.append(RiskAction(
                position_id=position_id,
                mission_id=mission_id,
                asset=asset,
                direction=direction,
                size=size,
                reason="TRAILING_STOP",
                current_price=current_price,
                trigger_price=trailing_stop_price or current_price,
            ))
            logger.warning(
                "RISK: Trailing stop triggered",
                mission_id=mission_id, asset=asset, direction=direction,
                current_price=current_price, trailing_stop=trailing_stop_price,
                peak=new_highest if direction == "LONG" else new_lowest,
            )
            continue

        # Priority 4: Take profit
        if tp_price > 0 and check_take_profit(direction, current_price, tp_price):
            actions.append(RiskAction(
                position_id=position_id,
                mission_id=mission_id,
                asset=asset,
                direction=direction,
                size=size,
                reason="TAKE_PROFIT",
                current_price=current_price,
                trigger_price=tp_price,
            ))
            logger.info(
                "RISK: Take profit triggered",
                mission_id=mission_id, asset=asset, direction=direction,
                current_price=current_price, tp_price=tp_price,
            )
            continue

        # Priority 5: Funding exit
        if margin_used > 0 and check_funding_exit(funding_paid, margin_used, profile["max_funding_cost_pct"]):
            actions.append(RiskAction(
                position_id=position_id,
                mission_id=mission_id,
                asset=asset,
                direction=direction,
                size=size,
                reason="FUNDING_EXIT",
                current_price=current_price,
                trigger_price=0,
            ))
            logger.warning(
                "RISK: Funding cost exit triggered",
                mission_id=mission_id, asset=asset,
                funding_paid=funding_paid, margin_used=margin_used,
                cost_pct=abs(funding_paid) / margin_used * 100,
            )
            continue

    # Mission-level drawdown kill switch
    kill_switch = False
    if initial_capital > 0 and account_value > 0:
        if check_mission_drawdown(account_value, initial_capital, profile["max_drawdown_pct"]):
            kill_switch = True
            drawdown_pct = ((initial_capital - account_value) / initial_capital) * 100
            logger.error(
                "RISK: MISSION DRAWDOWN KILL SWITCH",
                mission_id=mission_id,
                initial_capital=initial_capital,
                current_value=account_value,
                drawdown_pct=round(drawdown_pct, 2),
                threshold=profile["max_drawdown_pct"],
                risk_level=mission.get("risk_level"),
            )

            # If kill switch, add all remaining HL positions that aren't already in actions
            closed_assets = {a.asset for a in actions}
            for hl_pos in hl_positions:
                if hl_pos["asset"] not in closed_assets:
                    size = abs(hl_pos.get("size", 0))
                    if size > 0:
                        coin = hl_pos["asset"].replace("-USD", "")
                        price = cached_prices.get(coin, {}).get("markPx", hl_pos.get("entry_price", 0))
                        db_pos = db_pos_by_asset.get(hl_pos["asset"], {})
                        actions.append(RiskAction(
                            position_id=db_pos.get("id", f"hl_{hl_pos['asset']}_{mission_id}"),
                            mission_id=mission_id,
                            asset=hl_pos["asset"],
                            direction=hl_pos["direction"],
                            size=size,
                            reason="DRAWDOWN_KILL",
                            current_price=price,
                            trigger_price=0,
                        ))

    return actions, kill_switch


# ==================
# Executor
# ==================

async def execute_risk_closes(
    actions: List[RiskAction],
    mission: Dict[str, Any],
    hl_client,
    wallet_bridge,
) -> Dict[str, Any]:
    """
    Execute forced close orders for triggered risk actions.

    Args:
        actions: List of RiskAction to execute
        mission: Mission dict
        hl_client: HyperliquidClient instance
        wallet_bridge: TurnkeyBridge instance

    Returns:
        Summary dict with counts
    """
    from app.services.database import close_position, record_agent_audit
    from app.services.execution_queue import get_redis

    settings = get_settings()
    redis = await get_redis()

    closed = 0
    failed = 0
    mission_id = mission["id"]

    for action in actions:
        try:
            # Reverse direction to close
            is_buy = action.direction != "LONG"  # Buy to close SHORT, sell to close LONG

            if settings.dry_run:
                logger.info(
                    "DRY RUN: Would force-close position",
                    mission_id=mission_id,
                    asset=action.asset,
                    direction=action.direction,
                    size=action.size,
                    reason=action.reason,
                    current_price=action.current_price,
                    trigger_price=action.trigger_price,
                )
                order_result = {"success": True, "tx_hash": None}
            else:
                # Build reduce-only market order
                typed_data = hl_client.build_eip712_order(
                    asset=action.asset,
                    is_buy=is_buy,
                    size=action.size,
                    price=action.current_price,
                    reduce_only=True,
                )

                # Sign with per-mission agent key
                sign_result = await wallet_bridge.sign_with_agent_key(
                    mission_id=mission_id,
                    typed_data=typed_data,
                )

                if not sign_result.get("success"):
                    logger.error(
                        "Risk close signing failed",
                        mission_id=mission_id,
                        asset=action.asset,
                        reason=action.reason,
                        error=sign_result.get("error"),
                    )
                    failed += 1
                    continue

                order_result = await hl_client.place_order_eip712(
                    typed_data=typed_data,
                    signature=sign_result["signature"],
                )

            if order_result.get("success"):
                closed += 1

                # Calculate PnL for DB close
                # Simple PnL: (exit - entry) * qty for LONG, (entry - exit) * qty for SHORT
                entry_price = action.trigger_price if action.reason in ("STOP_LOSS", "TAKE_PROFIT") else action.current_price
                # We don't have exact entry price here, so use current price for realized PnL
                # The DB close_position just records what we have

                # Close in DB (if we have a real position ID)
                if not action.position_id.startswith("hl_"):
                    await close_position(
                        position_id=action.position_id,
                        close_price=action.current_price,
                        realized_pnl=0,  # PnL sync will update from HL state
                        close_reason=action.reason,
                    )

                # Clean up trailing stop state
                await delete_trailing_state(redis, action.position_id)

                # Audit log
                await record_agent_audit(
                    node="risk_manager",
                    action=f"force_close_{action.reason.lower()}",
                    mission_id=mission_id,
                    asset=action.asset,
                    decision={
                        "reason": action.reason,
                        "direction": action.direction,
                        "size": action.size,
                        "current_price": action.current_price,
                        "trigger_price": action.trigger_price,
                        "risk_level": mission.get("risk_level"),
                    },
                    reasoning=f"Risk manager force-closed {action.direction} {action.asset}: {action.reason}",
                    dry_run=settings.dry_run,
                )

                logger.warning(
                    "RISK CLOSE EXECUTED",
                    mission_id=mission_id,
                    asset=action.asset,
                    reason=action.reason,
                    direction=action.direction,
                    size=action.size,
                    dry_run=settings.dry_run,
                )
            else:
                failed += 1
                logger.error(
                    "Risk close order failed",
                    mission_id=mission_id,
                    asset=action.asset,
                    reason=action.reason,
                    error=order_result.get("error"),
                )

        except Exception as e:
            failed += 1
            logger.error(
                "Risk close exception",
                mission_id=mission_id,
                asset=action.asset,
                reason=action.reason,
                error=str(e),
                exc_info=True,
            )

    return {
        "mission_id": mission_id,
        "closed": closed,
        "failed": failed,
        "total_actions": len(actions),
    }

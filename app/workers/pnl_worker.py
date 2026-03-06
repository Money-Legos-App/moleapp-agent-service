"""
PnL Worker (Fan-Out)

Processes per-mission PnL updates using:
- Cached market prices from Redis (0ms, no API call)
- Single clearinghouseState call per mission (1 API call)

This replaces the old sync loop that made 2-3 API calls per mission.
"""

import json
from datetime import datetime

import structlog

logger = structlog.get_logger(__name__)

MARKET_PRICES_KEY = "agent:market:prices"
MARKET_PRICES_UPDATED_KEY = "agent:market:prices:updated_at"


async def update_pnl_for_mission(
    ctx: dict,
    mission_id: str,
    user_address: str,
    initial_capital: float,
) -> dict:
    """
    arq task: Update PnL for a single mission.

    1. Read cached prices from Redis (free)
    2. Call get_clearinghouse_state() once (1 HL API call)
    3. Update positions and mission totals in DB
    """
    from app.services.hyperliquid import HyperliquidClient
    from app.tasks.pnl_sync import _update_position, _update_mission_pnl
    from app.config import get_settings

    if not user_address:
        return {"status": "skipped", "reason": "no_wallet_address"}

    redis = ctx.get("redis")
    rate_limiter = ctx.get("rate_limiter")
    settings = get_settings()

    # 1. Read cached market prices from Redis
    cached_prices = {}
    if redis:
        raw_prices = await redis.hgetall(MARKET_PRICES_KEY)
        for coin, data_str in raw_prices.items():
            try:
                cached_prices[coin] = json.loads(data_str)
            except (json.JSONDecodeError, TypeError):
                pass

        # Check staleness
        updated_at = await redis.get(MARKET_PRICES_UPDATED_KEY)
        if updated_at:
            try:
                age = (datetime.utcnow() - datetime.fromisoformat(updated_at)).total_seconds()
                if age > settings.market_price_cache_ttl_seconds:
                    logger.warning(
                        "Market price cache is stale",
                        age_seconds=age,
                        mission_id=mission_id,
                    )
            except (ValueError, TypeError):
                pass

    # 2. Single API call: get positions + account value
    # Rate limiting is handled at the client level (HyperliquidRateLimiter)
    hl_client = HyperliquidClient()
    try:
        state = await hl_client.get_clearinghouse_state(user_address)
    finally:
        await hl_client.close()

    positions = state["positions"]
    account = state["account"]
    total_value = account["account_value"]

    # 4. Update each position using cached prices for current_price
    updated_count = 0
    for pos in positions:
        asset = pos["asset"]
        coin = asset.replace("-USD", "")

        # Use cached mark price if available, otherwise use HL-reported data
        if coin in cached_prices:
            current_price = cached_prices[coin].get("markPx", pos["entry_price"])
        else:
            current_price = pos["entry_price"]

        # Recalculate unrealized PnL with latest cached price
        if pos["direction"] == "LONG":
            unrealized_pnl = (current_price - pos["entry_price"]) * pos["size"]
        else:
            unrealized_pnl = (pos["entry_price"] - current_price) * pos["size"]

        await _update_position(
            mission_id=mission_id,
            asset=asset,
            current_price=current_price,
            unrealized_pnl=unrealized_pnl,
            funding_paid=pos.get("funding_paid", 0),
            margin_used=pos.get("margin_used", 0),
            liquidation_price=pos.get("liquidation_price"),
        )
        updated_count += 1

    # 5. Update mission totals
    total_pnl = total_value - initial_capital if initial_capital > 0 else 0

    await _update_mission_pnl(
        mission_id=mission_id,
        current_value=total_value,
        total_pnl=total_pnl,
    )

    logger.debug(
        "PnL updated for mission",
        mission_id=mission_id,
        positions=updated_count,
        total_value=total_value,
        total_pnl=total_pnl,
    )

    return {
        "status": "success",
        "positions_updated": updated_count,
        "total_value": total_value,
        "total_pnl": total_pnl,
    }

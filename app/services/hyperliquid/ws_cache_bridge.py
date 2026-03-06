"""
Bridge between WebSocket feed and Redis market price cache.

Converts allMids WS messages into the same Redis hash format
used by refresh_market_prices(), so PnL workers see no difference.

WS allMids provides only mid prices (markPx). Other fields like
funding, OI, and volume remain from the last REST refresh.
"""

import json
from datetime import datetime

import structlog
from redis.asyncio import Redis

logger = structlog.get_logger(__name__)

MARKET_PRICES_KEY = "agent:market:prices"
MARKET_PRICES_UPDATED_KEY = "agent:market:prices:updated_at"
WS_HEALTH_KEY = "agent:ws:healthy"


async def update_redis_from_ws_mids(mids: dict, redis: Redis) -> None:
    """
    Write allMids WS data into the same Redis hash that
    refresh_market_prices() writes to.

    allMids returns { "BTC": "95123.5", "ETH": "2645.1", ... }
    We update only the markPx field. Other fields (funding, OI, volume)
    remain from the last REST refresh.
    """
    if not mids:
        return

    pipe = redis.pipeline()

    for coin, mid_str in mids.items():
        try:
            mid_price = float(mid_str)
            # Read existing entry and update markPx, or create minimal entry
            existing_raw = await redis.hget(MARKET_PRICES_KEY, coin)
            if existing_raw:
                data = json.loads(existing_raw)
                data["markPx"] = mid_price
            else:
                data = {
                    "markPx": mid_price,
                    "funding": 0,
                    "openInterest": 0,
                    "dayNtlVlm": 0,
                    "prevDayPx": 0,
                }
            pipe.hset(MARKET_PRICES_KEY, coin, json.dumps(data))
        except (ValueError, TypeError, json.JSONDecodeError):
            continue

    pipe.set(MARKET_PRICES_UPDATED_KEY, datetime.utcnow().isoformat())
    # WS health key: TTL 60s, self-clears if WS dies
    pipe.set(WS_HEALTH_KEY, "1", ex=60)
    await pipe.execute()

"""
Market Data Worker (Singleton)

Fetches ALL market prices from Hyperliquid in a single API call,
caches them in Redis, then fans out per-mission PnL update jobs
to the arq queue.

Also caches candle-based technical summaries (RSI, EMA, ATR) in Redis
so the trading cycle reads pre-computed data instead of making 30 HTTP
requests per cycle.

This replaces the old pattern of N API calls (one per mission)
with 1 API call + N Redis reads.
"""

import asyncio
import json
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List

import structlog

logger = structlog.get_logger(__name__)

# Redis keys
MARKET_PRICES_KEY = "agent:market:prices"
MARKET_PRICES_UPDATED_KEY = "agent:market:prices:updated_at"
CANDLE_SUMMARY_PREFIX = "agent:candles:summary:"
CANDLE_SUMMARY_UPDATED_KEY = "agent:candles:updated_at"


async def refresh_market_prices() -> Dict[str, Any]:
    """
    Fetch market prices. Prefers WebSocket cache if healthy;
    falls back to REST API call if WS is stale or disconnected.

    Returns:
        Dict of coin -> price data
    """
    from app.services.hyperliquid import HyperliquidClient
    from app.services.execution_queue import get_redis
    from app.services.hyperliquid.ws_cache_bridge import WS_HEALTH_KEY

    redis = await get_redis()

    # Check if WebSocket feed is keeping the cache fresh
    ws_healthy = await redis.get(WS_HEALTH_KEY)
    if ws_healthy:
        # WS is healthy -- read the cache it already populated (0 API calls)
        raw_prices = await redis.hgetall(MARKET_PRICES_KEY)
        prices = {}
        for coin, data_str in raw_prices.items():
            try:
                prices[coin] = json.loads(data_str)
            except (json.JSONDecodeError, TypeError):
                pass
        if prices:
            logger.debug("Using WS-cached prices", assets=len(prices))
            return prices

    # Fallback: REST API call (WS unavailable or cache empty)
    logger.info("WS cache stale or unavailable, falling back to REST")
    hl_client = HyperliquidClient()
    try:
        prices = await hl_client.get_all_market_prices()

        # Write all prices to a Redis hash (one field per coin)
        if prices:
            pipe = redis.pipeline()
            for coin, data in prices.items():
                pipe.hset(MARKET_PRICES_KEY, coin, json.dumps(data))
            pipe.set(MARKET_PRICES_UPDATED_KEY, datetime.utcnow().isoformat())
            await pipe.execute()

        logger.info(
            "Market prices cached in Redis (REST fallback)",
            assets_cached=len(prices),
        )
        return prices

    finally:
        await hl_client.close()


async def refresh_candle_summaries() -> Dict[str, str]:
    """
    Fetch candles for all active assets (semaphore-batched), compute
    technical summaries, and cache in Redis.

    Runs every 5 min as a background task so the trading cycle reads
    pre-computed data from Redis (0 HTTP requests in the hot path).

    Returns:
        Dict of asset -> summary string
    """
    from app.services.hyperliquid import HyperliquidClient
    from app.services.execution_queue import get_redis
    from app.config import get_settings

    settings = get_settings()
    redis = await get_redis()

    # Resolve active asset list (dynamic > static)
    active_assets: List[str] = list(settings.allowed_assets)
    if settings.dynamic_asset_rotation_enabled:
        try:
            dynamic_raw = await redis.get("agent:dynamic:allowed_assets")
            if dynamic_raw:
                dynamic_list = json.loads(
                    dynamic_raw if isinstance(dynamic_raw, str) else dynamic_raw.decode()
                )
                if dynamic_list:
                    active_assets = dynamic_list
        except Exception:
            pass

    hl_client = HyperliquidClient()
    semaphore = asyncio.Semaphore(5)  # max 5 concurrent HL requests
    summaries: Dict[str, str] = {}

    async def _fetch_one(asset: str):
        coin = asset.replace("-USD", "")
        async with semaphore:
            try:
                summary = await hl_client.get_multi_timeframe_analysis(coin)
                return asset, summary
            except Exception as e:
                logger.warning("Candle summary fetch failed", coin=coin, error=str(e))
                return asset, None

    try:
        results = await asyncio.gather(
            *[_fetch_one(a) for a in active_assets],
            return_exceptions=True,
        )

        pipe = redis.pipeline()
        cached = 0
        for result in results:
            if isinstance(result, Exception):
                continue
            asset, summary = result
            if summary:
                summaries[asset] = summary
                pipe.set(
                    f"{CANDLE_SUMMARY_PREFIX}{asset}",
                    summary,
                    ex=600,  # 10 min TTL (refresh runs every 5 min)
                )
                cached += 1

        pipe.set(CANDLE_SUMMARY_UPDATED_KEY, datetime.utcnow().isoformat(), ex=600)
        await pipe.execute()

        logger.info(
            "Candle summaries cached",
            assets_cached=cached,
            total_assets=len(active_assets),
        )

    finally:
        await hl_client.close()

    return summaries


async def fan_out_pnl_jobs() -> int:
    """
    Enqueue per-mission PnL jobs in staggered batches to prevent
    thundering herd on Hyperliquid's API.

    Jobs are shuffled, batched, and enqueued with random jitter so
    the arq workers process them over time rather than all at once.

    Returns:
        Number of jobs enqueued
    """
    from app.services.execution_queue import get_arq_pool
    from app.tasks.pnl_sync import _get_active_missions_with_wallets
    from app.config import get_settings

    settings = get_settings()
    batch_size = settings.pnl_fanout_batch_size
    jitter_max_ms = settings.pnl_fanout_jitter_max_ms

    missions = await _get_active_missions_with_wallets()

    if not missions:
        logger.info("No active missions for PnL fan-out")
        return 0

    arq_pool = await get_arq_pool()
    enqueued = 0

    # Shuffle to avoid always hitting the same missions first
    random.shuffle(missions)

    for i in range(0, len(missions), batch_size):
        batch = missions[i:i + batch_size]

        for mission in batch:
            try:
                # Add random jitter per job to stagger execution
                jitter_ms = random.randint(0, jitter_max_ms)
                await arq_pool.enqueue_job(
                    "update_pnl_for_mission",
                    mission_id=mission["id"],
                    user_address=mission.get("user_wallet_address", ""),
                    initial_capital=float(mission.get("initial_capital", 0)),
                    _defer_by=timedelta(milliseconds=jitter_ms),
                )
                enqueued += 1
            except Exception as e:
                logger.error(
                    "Failed to enqueue PnL job",
                    mission_id=mission["id"],
                    error=str(e),
                )

        # Wait between batches to spread load
        if i + batch_size < len(missions):
            await asyncio.sleep(1.0)

    logger.info(
        "PnL jobs fanned out with jitter",
        jobs_enqueued=enqueued,
        batch_size=batch_size,
        jitter_max_ms=jitter_max_ms,
    )
    return enqueued

"""
Dynamic Asset Rotation

Daily task that ranks HL assets by turnover ratio (volume / notional OI)
and updates the monitored asset list. Position-safe: always includes
assets with open positions to prevent stranded trades.

Runs at 01:00 UTC (after daily close chaos settles).
"""

import json
from typing import Dict, List, Set

import structlog

from app.config import get_settings

logger = structlog.get_logger(__name__)

DYNAMIC_ASSETS_KEY = "agent:dynamic:allowed_assets"
MANDATORY_ASSETS = {"BTC-USD", "ETH-USD"}


async def rotate_assets() -> Dict:
    """
    Rank all HL assets by turnover ratio and build the optimal trading list.

    Array construction order:
    1. mandatory = {BTC-USD, ETH-USD}
    2. open_positions = assets with current open positions (prevents stranding)
    3. remaining_slots = max_count - len(mandatory | open_positions)
    4. candidates = ranked by turnover_ratio, filtered by $15M volume floor
    5. final_list = mandatory | open_positions | top_N(candidates)
    """
    from app.services.execution_queue import get_redis
    from app.services.database import get_all_open_positions_with_risk

    settings = get_settings()
    redis = await get_redis()

    # Step 1: Mandatory anchors
    selected: Set[str] = set(MANDATORY_ASSETS)

    # Step 2: Force-include assets with open positions
    open_position_assets: Set[str] = set()
    try:
        positions = await get_all_open_positions_with_risk()
        for pos in positions:
            asset = pos.get("asset", "")
            if asset:
                open_position_assets.add(asset)
                selected.add(asset)
    except Exception as e:
        logger.warning("Asset rotation: failed to fetch open positions", error=str(e))

    # Step 3: Fetch fresh market data via REST (not Redis cache, which may
    # only have WS mid-prices without OI/volume needed for ranking)
    from app.services.hyperliquid import HyperliquidClient

    hl_client = HyperliquidClient()
    try:
        all_prices = await hl_client.get_all_market_prices()
    except Exception as e:
        logger.warning("Asset rotation: REST fetch failed, keeping static list", error=str(e))
        return {"status": "rest_error", "assets": list(settings.allowed_assets)}
    finally:
        await hl_client.close()

    if not all_prices:
        logger.warning("Asset rotation: empty REST response, keeping static list")
        return {"status": "no_data", "assets": list(settings.allowed_assets)}

    # Step 4: Rank all assets by turnover ratio
    candidates = []
    for coin, data in all_prices.items():
        asset = f"{coin}-USD"
        if asset in selected:
            continue  # already in mandatory/open positions

        mark_px = float(data.get("markPx", 0))
        open_interest = float(data.get("openInterest", 0))
        volume_24h = float(data.get("dayNtlVlm", 0))

        # Filter: minimum volume floor
        if volume_24h < settings.dynamic_asset_min_volume_24h:
            continue

        # Turnover ratio: high volume relative to OI = fast-moving asset
        notional_oi = mark_px * open_interest
        if notional_oi <= 0:
            continue

        turnover_ratio = volume_24h / notional_oi

        candidates.append({
            "asset": asset,
            "turnover_ratio": turnover_ratio,
            "volume_24h": volume_24h,
            "notional_oi": notional_oi,
        })

    # Sort by turnover ratio (highest first)
    candidates.sort(key=lambda x: x["turnover_ratio"], reverse=True)

    # Step 5: Fill remaining slots
    remaining_slots = max(0, settings.dynamic_asset_max_count - len(selected))
    for candidate in candidates[:remaining_slots]:
        selected.add(candidate["asset"])

    final_list = sorted(selected)

    # Get previous list for diff logging
    prev_raw = await redis.get(DYNAMIC_ASSETS_KEY)
    prev_list = set()
    if prev_raw:
        try:
            prev_list = set(json.loads(prev_raw if isinstance(prev_raw, str) else prev_raw.decode()))
        except (json.JSONDecodeError, TypeError):
            pass

    added = selected - prev_list if prev_list else set()
    removed = prev_list - selected if prev_list else set()

    # Store in Redis with 25h TTL (survives until next rotation + buffer)
    await redis.set(
        DYNAMIC_ASSETS_KEY,
        json.dumps(final_list),
        ex=25 * 3600,
    )

    logger.info(
        "Asset rotation complete",
        total=len(final_list),
        mandatory=list(MANDATORY_ASSETS),
        open_positions=list(open_position_assets),
        added=list(added) if added else [],
        removed=list(removed) if removed else [],
        top_turnover=[
            f"{c['asset']}({c['turnover_ratio']:.2f})"
            for c in candidates[:5]
        ],
    )

    return {
        "status": "ok",
        "assets": final_list,
        "added": list(added),
        "removed": list(removed),
        "open_positions_protected": list(open_position_assets),
    }

"""
PnL Synchronization Tasks
Syncs position data and PnL from Hyperliquid
"""

from datetime import datetime
from typing import List, Dict, Any

import structlog

logger = structlog.get_logger(__name__)


async def sync_positions_from_hyperliquid() -> int:
    """
    DEPRECATED: Use update_pnl_for_mission arq task instead.

    This function makes 3 API calls per mission (get_positions + get_mark_prices +
    get_account_value). The arq fan-out approach (market_data_worker + pnl_worker)
    uses 1 REST call total + 1 call per mission via get_clearinghouse_state().

    Kept for backwards compatibility but no longer called by the scheduler.

    Returns:
        Number of positions updated
    """
    import warnings
    warnings.warn(
        "sync_positions_from_hyperliquid is deprecated. Use the arq fan-out pattern instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from app.services.hyperliquid import HyperliquidClient
    from app.services.wallet import TurnkeyBridge

    logger.info("Starting position sync from Hyperliquid")

    hl_client = HyperliquidClient()
    wallet_bridge = TurnkeyBridge()
    updated_count = 0

    try:
        # Get all active missions
        missions = await _get_active_missions_with_wallets()

        if not missions:
            logger.info("No active missions to sync")
            return 0

        for mission in missions:
            try:
                user_address = mission.get("user_wallet_address")
                if not user_address:
                    continue

                # Fetch current positions from Hyperliquid
                positions = await hl_client.get_positions(user_address)

                # Fetch current prices
                prices = await hl_client.get_mark_prices()

                # Update each position
                for pos in positions:
                    asset = pos["asset"]
                    current_price = prices.get(asset.replace("-USD", ""), pos["entry_price"])

                    # Calculate unrealized PnL
                    if pos["direction"] == "LONG":
                        unrealized_pnl = (current_price - pos["entry_price"]) * pos["size"]
                    else:
                        unrealized_pnl = (pos["entry_price"] - current_price) * pos["size"]

                    # Update position in database
                    await _update_position(
                        mission_id=mission["id"],
                        asset=asset,
                        current_price=current_price,
                        unrealized_pnl=unrealized_pnl,
                        funding_paid=pos.get("funding_paid", 0),
                        margin_used=pos.get("margin_used", 0),
                        liquidation_price=pos.get("liquidation_price"),
                    )
                    updated_count += 1

                # Get account value
                account = await hl_client.get_account_value(user_address)
                total_value = account.get("account_value", 0)

                # Calculate total PnL
                initial_capital = float(mission.get("initial_capital", 0))
                total_pnl = total_value - initial_capital if initial_capital > 0 else 0

                # Update mission totals
                await _update_mission_pnl(
                    mission_id=mission["id"],
                    current_value=total_value,
                    total_pnl=total_pnl,
                )

            except Exception as e:
                logger.error(
                    "Error syncing mission positions",
                    mission_id=mission["id"],
                    error=str(e),
                )
                continue

    finally:
        await hl_client.close()
        await wallet_bridge.close()

    logger.info("Position sync completed", positions_updated=updated_count)
    return updated_count


async def create_daily_snapshots() -> int:
    """
    Create daily PnL snapshots for all active missions.

    Uses a single get_clearinghouse_state() call per mission (1 API call)
    instead of separate get_account_value() + get_positions() (2 API calls).

    Returns:
        Number of snapshots created
    """
    from app.services.hyperliquid import HyperliquidClient
    from app.services.wallet import TurnkeyBridge

    logger.info("Creating daily PnL snapshots")

    hl_client = HyperliquidClient()
    wallet_bridge = TurnkeyBridge()
    snapshot_count = 0

    try:
        missions = await _get_active_missions_with_wallets()

        for mission in missions:
            try:
                user_address = mission.get("user_wallet_address")
                if not user_address:
                    continue

                # Single API call: positions + account value combined
                state = await hl_client.get_clearinghouse_state(user_address)
                positions = state["positions"]
                account = state["account"]
                total_value = account["account_value"]

                # Calculate PnL
                initial_capital = float(mission.get("initial_capital", 0))
                total_pnl = total_value - initial_capital if initial_capital > 0 else 0

                # Unrealized/realized split
                unrealized_pnl = sum(p.get("unrealized_pnl", 0) for p in positions)
                realized_pnl = total_pnl - unrealized_pnl

                # Record snapshot via wallet-service
                await wallet_bridge.record_pnl_snapshot(
                    mission_id=mission["id"],
                    total_value=total_value,
                    total_pnl=total_pnl,
                    unrealized_pnl=unrealized_pnl,
                    realized_pnl=realized_pnl,
                )
                snapshot_count += 1

            except Exception as e:
                logger.error(
                    "Error creating snapshot for mission",
                    mission_id=mission["id"],
                    error=str(e),
                )
                continue

    finally:
        await hl_client.close()
        await wallet_bridge.close()

    logger.info("Daily snapshots created", count=snapshot_count)
    return snapshot_count


async def _get_active_missions_with_wallets() -> List[Dict[str, Any]]:
    """Fetch active missions with wallet addresses."""
    from app.services.database import get_active_missions_with_wallets
    try:
        return await get_active_missions_with_wallets()
    except Exception as e:
        logger.error("Failed to fetch missions for PnL sync", error=str(e))
        return []


async def _update_position(
    mission_id: str,
    asset: str,
    current_price: float,
    unrealized_pnl: float,
    funding_paid: float,
    margin_used: float,
    liquidation_price: float = None,
) -> None:
    """Update position in database."""
    from app.services.database import update_position, get_open_positions
    try:
        positions = await get_open_positions(mission_id)
        for pos in positions:
            if pos["asset"] == asset:
                await update_position(
                    position_id=pos["id"],
                    current_price=current_price,
                    unrealized_pnl=unrealized_pnl,
                    funding_paid=funding_paid,
                )
                return
        logger.debug("No matching position found to update", mission_id=mission_id, asset=asset)
    except Exception as e:
        logger.error("Failed to update position", error=str(e), mission_id=mission_id, asset=asset)


async def _update_mission_pnl(
    mission_id: str,
    current_value: float,
    total_pnl: float,
) -> None:
    """Update mission PnL totals."""
    from app.services.database import update_mission_pnl
    try:
        await update_mission_pnl(
            mission_id=mission_id,
            current_value=current_value,
            total_pnl=total_pnl,
        )
    except Exception as e:
        logger.error("Failed to update mission PnL", error=str(e), mission_id=mission_id)

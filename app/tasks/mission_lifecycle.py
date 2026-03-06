"""
Mission Lifecycle Tasks
Handles mission expiry, completion, and force-close scenarios
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List

import structlog

logger = structlog.get_logger(__name__)


async def check_mission_expiry() -> Dict[str, int]:
    """
    Check for missions approaching expiry and handle accordingly.

    Actions:
    - Notify users when mission has < 3 days remaining
    - Begin position wind-down when < 2 days remaining
    - Force close all positions when mission ends
    - Mark completed missions

    Returns:
        Dictionary with counts of actions taken
    """
    logger.info("Checking mission expiry")

    result = {
        "ending_soon": 0,
        "force_closed": 0,
        "completed": 0,
        "notified": 0,
    }

    try:
        # Get missions ending soon (within 3 days)
        missions = await _get_missions_ending_soon(days=3)

        for mission in missions:
            mission_id = mission["id"]
            ends_at = mission.get("ends_at")

            if not ends_at:
                continue

            if isinstance(ends_at, str):
                ends_at = datetime.fromisoformat(ends_at.replace("Z", "+00:00"))

            days_remaining = (ends_at.replace(tzinfo=None) - datetime.utcnow()).days

            try:
                if days_remaining <= 0:
                    # Mission has ended - force close and complete
                    await _force_close_mission(mission_id)
                    result["force_closed"] += 1
                    result["completed"] += 1
                    logger.info(
                        "Mission force closed and completed",
                        mission_id=mission_id,
                    )

                elif days_remaining <= 2:
                    # Last 2 days - no new positions, wind down existing
                    result["ending_soon"] += 1
                    logger.info(
                        "Mission in wind-down period",
                        mission_id=mission_id,
                        days_remaining=days_remaining,
                    )

                elif days_remaining <= 3:
                    # Notify user
                    await _notify_mission_ending(mission_id, days_remaining)
                    result["notified"] += 1
                    result["ending_soon"] += 1

            except Exception as e:
                logger.error(
                    "Error processing mission expiry",
                    mission_id=mission_id,
                    error=str(e),
                )
                continue

    except Exception as e:
        logger.error("Mission expiry check failed", error=str(e))

    if any(result.values()):
        logger.info("Mission expiry check completed", **result)

    return result


async def _get_missions_ending_soon(days: int) -> List[Dict[str, Any]]:
    """Get missions ending within specified days."""
    from app.services.database import get_missions_ending_soon
    try:
        return await get_missions_ending_soon(days=days)
    except Exception as e:
        logger.error("Failed to fetch missions ending soon", error=str(e))
        return []


async def _force_close_mission(mission_id: str) -> None:
    """
    Force close all positions and mark mission as completed.

    Steps:
    1. Get all open positions
    2. Submit market sell orders for all positions
    3. Update mission status to COMPLETED
    """
    from app.services.hyperliquid import HyperliquidClient
    from app.services.wallet import TurnkeyBridge

    logger.info("Force closing mission", mission_id=mission_id)

    # Audit: Log start of force close
    await _audit_log(
        node="lifecycle",
        action="force_close_started",
        mission_id=mission_id,
        reasoning="Mission expired, initiating force close of all positions",
    )

    hl_client = HyperliquidClient()
    wallet_bridge = TurnkeyBridge()

    positions_closed = 0
    positions_failed = 0

    try:
        # Get mission details
        mission = await wallet_bridge.get_mission_details(mission_id)
        if not mission:
            logger.error("Mission not found for force close", mission_id=mission_id)
            return

        user_address = mission.get("userWalletAddress")
        if not user_address:
            logger.error("User wallet address not found", mission_id=mission_id)
            return

        # Get all open positions
        positions = await hl_client.get_positions(user_address)

        if positions:
            # Build close orders for all positions
            close_orders = []
            for pos in positions:
                is_buy = pos["direction"] == "SHORT"  # Buy to close short, sell to close long

                order_payload = hl_client.build_order_payload(
                    asset=pos["asset"],
                    is_buy=is_buy,
                    size=pos["size"],
                    price=None,  # Market order
                    reduce_only=True,
                    order_type="market",
                )

                close_orders.append({
                    "missionId": mission_id,
                    "payload": order_payload,
                })

            # Batch sign and submit
            if close_orders:
                signed = await wallet_bridge.batch_sign_trades(close_orders)

                for i, (order, sign_result) in enumerate(zip(close_orders, signed)):
                    if sign_result.get("success"):
                        try:
                            await hl_client.place_order(sign_result)
                            positions_closed += 1
                            logger.info(
                                "Force closed position",
                                mission_id=mission_id,
                                asset=positions[i]["asset"],
                            )
                        except Exception as e:
                            positions_failed += 1
                            logger.error(
                                "Failed to place force close order",
                                mission_id=mission_id,
                                error=str(e),
                            )
                    else:
                        positions_failed += 1

        # Application-level audit log
        logger.info(
            "AUDIT lifecycle force_close positions result",
            mission_id=mission_id,
            total_positions=len(positions) if positions else 0,
            positions_closed=positions_closed,
            positions_failed=positions_failed,
        )

        # DB audit log
        await _audit_log(
            node="lifecycle",
            action="force_close_positions",
            mission_id=mission_id,
            decision={
                "total_positions": len(positions) if positions else 0,
                "positions_closed": positions_closed,
                "positions_failed": positions_failed,
            },
            reasoning=f"Force closed {positions_closed}/{len(positions) if positions else 0} positions",
            success=positions_failed == 0,
        )

        # Withdraw remaining funds from Hyperliquid back to smart wallet
        try:
            # Get user's remaining balance on HL
            account_state = await hl_client.get_positions(user_address)
            # The account value after closing positions is the withdrawable amount
            withdraw_amount = mission.get("currentValue") or mission.get("initialCapital", "0")

            signed_withdrawal = await wallet_bridge.withdraw_from_hyperliquid(
                mission_id=mission_id,
                amount=str(withdraw_amount),
            )

            if signed_withdrawal.get("success"):
                submit_result = await hl_client.submit_withdrawal(signed_withdrawal)
                if submit_result.get("success"):
                    logger.info("Funds withdrawn from HL", mission_id=mission_id, amount=withdraw_amount)

                    # Audit: Log successful withdrawal
                    await _audit_log(
                        node="lifecycle",
                        action="withdrawal_completed",
                        mission_id=mission_id,
                        decision={"amount": str(withdraw_amount)},
                        reasoning=f"Withdrew {withdraw_amount} USDC from Hyperliquid",
                        success=True,
                    )
                else:
                    logger.error("Failed to submit withdrawal to HL", mission_id=mission_id, error=submit_result.get("error"))

                    # Audit: Log failed withdrawal
                    await _audit_log(
                        node="lifecycle",
                        action="withdrawal_failed",
                        mission_id=mission_id,
                        decision={"amount": str(withdraw_amount), "stage": "submission"},
                        success=False,
                        error_message=submit_result.get("error"),
                    )
            else:
                logger.error("Failed to sign withdrawal", mission_id=mission_id, error=signed_withdrawal.get("error"))

                await _audit_log(
                    node="lifecycle",
                    action="withdrawal_failed",
                    mission_id=mission_id,
                    decision={"amount": str(withdraw_amount), "stage": "signing"},
                    success=False,
                    error_message=signed_withdrawal.get("error"),
                )
        except Exception as e:
            logger.error("Withdrawal step failed (positions already closed)", mission_id=mission_id, error=str(e))

            await _audit_log(
                node="lifecycle",
                action="withdrawal_failed",
                mission_id=mission_id,
                success=False,
                error_message=str(e),
            )

        # Update mission status to COMPLETED
        await wallet_bridge.update_mission_status(
            mission_id=mission_id,
            status="COMPLETED",
            metadata={"completedAt": datetime.utcnow().isoformat()},
        )

        # Audit: Log mission completion
        await _audit_log(
            node="lifecycle",
            action="mission_completed",
            mission_id=mission_id,
            decision={
                "positions_closed": positions_closed,
                "positions_failed": positions_failed,
                "final_status": "COMPLETED",
            },
            reasoning=f"Mission force closed and completed. {positions_closed} positions closed.",
            success=True,
        )

        logger.info(
            "Mission force close completed",
            mission_id=mission_id,
            positions_closed=len(positions) if positions else 0,
        )

    finally:
        await hl_client.close()
        await wallet_bridge.close()


async def _notify_mission_ending(mission_id: str, days_remaining: int) -> None:
    """
    Send notification to user about mission ending soon.
    """
    # TODO: Call notification-service
    logger.info(
        "Notification: Mission ending soon",
        mission_id=mission_id,
        days_remaining=days_remaining,
    )


async def complete_mission(mission_id: str) -> Dict[str, Any]:
    """
    Complete a mission normally (not force close).

    Called when user voluntarily ends mission or mission reaches end date
    with all positions already closed.
    """
    from app.services.wallet import TurnkeyBridge

    logger.info("Completing mission", mission_id=mission_id)

    wallet_bridge = TurnkeyBridge()

    try:
        # Update status
        result = await wallet_bridge.update_mission_status(
            mission_id=mission_id,
            status="COMPLETED",
        )

        # Audit: Log normal mission completion
        await _audit_log(
            node="lifecycle",
            action="mission_completed",
            mission_id=mission_id,
            reasoning="Mission completed normally (user-initiated or all positions closed)",
            success=True,
        )

        logger.info("Mission completed", mission_id=mission_id)
        return {"success": True, "mission_id": mission_id}

    finally:
        await wallet_bridge.close()


async def _audit_log(**kwargs) -> None:
    """Write an audit log entry, swallowing errors to avoid disrupting the pipeline."""
    try:
        from app.services.database import record_agent_audit
        await record_agent_audit(**kwargs)
    except Exception as e:
        logger.warning("Failed to write audit log", error=str(e))

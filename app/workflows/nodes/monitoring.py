"""
Position Monitoring Node
Monitors open positions for risk events and exit signals
"""

from datetime import datetime
from typing import Any, Dict, List

import structlog

from app.workflows.state import AgentState

logger = structlog.get_logger(__name__)


async def position_monitoring_node(state: AgentState) -> Dict[str, Any]:
    """
    Position Monitoring Node.

    Checks:
    1. Positions approaching liquidation
    2. Stop loss triggers
    3. Take profit triggers
    4. Position health alerts

    Args:
        state: Current workflow state

    Returns:
        Updated state with positions to close and alerts
    """
    from app.services.hyperliquid import HyperliquidClient
    from app.services.llm import DeepSeekClient

    logger.info("Starting position monitoring")

    errors = state.get("errors", [])
    positions_to_close: List[Dict[str, Any]] = []
    position_alerts: List[Dict[str, Any]] = []

    # Initialize services
    hl_client = HyperliquidClient()
    llm = DeepSeekClient()

    try:
        # Get active missions with open positions
        missions = await _get_missions_with_positions()

        if not missions:
            logger.info("No missions with positions to monitor")
            return {
                "positions_to_close": [],
                "position_alerts": [],
                "errors": errors,
                "completed_nodes": state.get("completed_nodes", []) + ["monitoring"],
            }

        for mission in missions:
            mission_id = mission["id"]
            user_address = mission.get("user_wallet_address")

            try:
                # Get current positions
                positions = await hl_client.get_positions(user_address)

                if not positions:
                    continue

                # Calculate mission context
                started_at = mission.get("started_at")
                if started_at:
                    if isinstance(started_at, str):
                        started_at = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                    mission_day = (datetime.utcnow() - started_at.replace(tzinfo=None)).days + 1
                else:
                    mission_day = 1

                duration_days = mission.get("duration_days", 30)
                days_remaining = duration_days - mission_day + 1

                for position in positions:
                    asset = position["asset"]
                    current_price = position.get("entry_price", 0)  # Would be updated from market

                    # Get current market price
                    market_data = await hl_client.get_market_data(asset)
                    current_price = market_data.get("price", current_price)

                    # Calculate unrealized PnL percentage
                    entry_price = position["entry_price"]
                    if entry_price > 0:
                        if position["direction"] == "LONG":
                            pnl_percent = ((current_price - entry_price) / entry_price) * 100
                        else:
                            pnl_percent = ((entry_price - current_price) / entry_price) * 100
                    else:
                        pnl_percent = 0

                    # Check for liquidation risk
                    liquidation_price = position.get("liquidation_price", 0)
                    if liquidation_price > 0:
                        if position["direction"] == "LONG":
                            distance_to_liq = ((current_price - liquidation_price) / current_price) * 100
                        else:
                            distance_to_liq = ((liquidation_price - current_price) / current_price) * 100

                        if distance_to_liq < 10:  # Within 10% of liquidation
                            position_alerts.append({
                                "mission_id": mission_id,
                                "type": "LIQUIDATION_WARNING",
                                "asset": asset,
                                "direction": position["direction"],
                                "distance_to_liquidation_percent": distance_to_liq,
                                "current_price": current_price,
                                "liquidation_price": liquidation_price,
                                "urgency": "high" if distance_to_liq < 5 else "medium",
                            })

                            # Force close if very close to liquidation
                            if distance_to_liq < 3:
                                positions_to_close.append({
                                    "mission_id": mission_id,
                                    "user_id": mission["user_id"],
                                    "action": f"EXIT_{position['direction']}",
                                    "asset": asset,
                                    "quantity": position["size"],
                                    "reason": "emergency_liquidation_risk",
                                })

                                # Audit: Log emergency liquidation close
                                await _audit_log(
                                    node="monitoring",
                                    action="emergency_exit",
                                    mission_id=mission_id,
                                    user_id=mission.get("user_id"),
                                    asset=asset,
                                    decision={
                                        "direction": position["direction"],
                                        "distance_to_liq_percent": distance_to_liq,
                                        "current_price": current_price,
                                        "liquidation_price": liquidation_price,
                                    },
                                    reasoning=f"Emergency close: {distance_to_liq:.1f}% from liquidation",
                                )

                                logger.warning(
                                    "Emergency close - liquidation risk",
                                    mission_id=mission_id,
                                    asset=asset,
                                    distance_to_liq=distance_to_liq,
                                )

                    # Check stop loss (if we had stored it)
                    stop_loss = mission.get("positions", {}).get(asset, {}).get("stop_loss_price")
                    if stop_loss:
                        if position["direction"] == "LONG" and current_price <= stop_loss:
                            positions_to_close.append({
                                "mission_id": mission_id,
                                "user_id": mission["user_id"],
                                "action": f"EXIT_{position['direction']}",
                                "asset": asset,
                                "quantity": position["size"],
                                "reason": "stop_loss_triggered",
                            })
                        elif position["direction"] == "SHORT" and current_price >= stop_loss:
                            positions_to_close.append({
                                "mission_id": mission_id,
                                "user_id": mission["user_id"],
                                "action": f"EXIT_{position['direction']}",
                                "asset": asset,
                                "quantity": position["size"],
                                "reason": "stop_loss_triggered",
                            })

                    # Check take profit
                    take_profit = mission.get("positions", {}).get(asset, {}).get("take_profit_price")
                    if take_profit:
                        if position["direction"] == "LONG" and current_price >= take_profit:
                            positions_to_close.append({
                                "mission_id": mission_id,
                                "user_id": mission["user_id"],
                                "action": f"EXIT_{position['direction']}",
                                "asset": asset,
                                "quantity": position["size"],
                                "reason": "take_profit_triggered",
                            })
                        elif position["direction"] == "SHORT" and current_price <= take_profit:
                            positions_to_close.append({
                                "mission_id": mission_id,
                                "user_id": mission["user_id"],
                                "action": f"EXIT_{position['direction']}",
                                "asset": asset,
                                "quantity": position["size"],
                                "reason": "take_profit_triggered",
                            })

                    # Use LLM for advanced exit analysis if significant profit/loss
                    if abs(pnl_percent) > 15:  # >15% move
                        position_context = {
                            "id": f"{mission_id}_{asset}",
                            "asset": asset,
                            "direction": position["direction"],
                            "entry_price": entry_price,
                            "current_price": current_price,
                            "unrealized_pnl_percent": pnl_percent,
                            "position_value": position["size"] * current_price,
                            "leverage": position.get("leverage", 1),
                            "hours_in_position": 0,  # Would calculate from position open time
                        }

                        mission_context = {
                            "mission_day": mission_day,
                            "duration_days": duration_days,
                            "days_remaining": days_remaining,
                            "total_pnl_percent": mission.get("total_pnl", 0),
                        }

                        market_context = {
                            "price_change_24h": market_data.get("price_change_24h", 0),
                            "volatility": market_data.get("volatility", 5),
                        }

                        exit_decision = await llm.analyze_position_exit(
                            position=position_context,
                            mission=mission_context,
                            market_data=market_context,
                        )

                        # Application-level audit log
                        logger.info(
                            "AUDIT monitoring LLM exit analysis",
                            mission_id=mission_id,
                            asset=asset,
                            direction=position["direction"],
                            pnl_percent=round(pnl_percent, 2),
                            should_exit=exit_decision.get("should_exit"),
                            exit_reason=exit_decision.get("exit_reason"),
                            urgency=exit_decision.get("urgency"),
                            reasoning=exit_decision.get("reasoning", "")[:200],
                            tokens_used=exit_decision.get("_audit_tokens"),
                        )

                        # DB audit log
                        await _audit_log(
                            node="monitoring",
                            action="llm_call",
                            mission_id=mission_id,
                            user_id=mission.get("user_id"),
                            asset=asset,
                            llm_prompt=exit_decision.get("_audit_prompt"),
                            llm_response=exit_decision.get("_audit_response"),
                            llm_model=exit_decision.get("_audit_model"),
                            llm_tokens=exit_decision.get("_audit_tokens"),
                            decision={
                                "should_exit": exit_decision.get("should_exit"),
                                "exit_reason": exit_decision.get("exit_reason"),
                                "urgency": exit_decision.get("urgency"),
                                "pnl_percent": pnl_percent,
                            },
                            reasoning=exit_decision.get("reasoning", ""),
                            metadata={"position": position_context, "mission": mission_context},
                            success="_audit_error" not in exit_decision,
                            error_message=exit_decision.get("_audit_error"),
                        )

                        if exit_decision.get("should_exit", False):
                            positions_to_close.append({
                                "mission_id": mission_id,
                                "user_id": mission["user_id"],
                                "action": f"EXIT_{position['direction']}",
                                "asset": asset,
                                "quantity": position["size"],
                                "reason": exit_decision.get("exit_reason", "llm_decision"),
                                "llm_reasoning": exit_decision.get("reasoning"),
                            })

                            # Audit: Log the exit decision separately
                            await _audit_log(
                                node="monitoring",
                                action="position_exit_decision",
                                mission_id=mission_id,
                                user_id=mission.get("user_id"),
                                asset=asset,
                                decision={
                                    "direction": position["direction"],
                                    "quantity": position["size"],
                                    "pnl_percent": pnl_percent,
                                    "exit_reason": exit_decision.get("exit_reason"),
                                },
                                reasoning=exit_decision.get("reasoning", ""),
                            )

                            logger.info(
                                "LLM exit decision",
                                mission_id=mission_id,
                                asset=asset,
                                reason=exit_decision.get("exit_reason"),
                            )

            except Exception as e:
                logger.error(
                    "Error monitoring mission positions",
                    mission_id=mission_id,
                    error=str(e),
                )
                errors.append(f"Monitoring error for mission {mission_id}: {str(e)}")

    finally:
        await hl_client.close()
        await llm.close()

    logger.info(
        "Position monitoring completed",
        positions_to_close=len(positions_to_close),
        alerts=len(position_alerts),
    )

    return {
        "positions_to_close": positions_to_close,
        "position_alerts": position_alerts,
        "errors": errors,
        "completed_nodes": state.get("completed_nodes", []) + ["monitoring"],
    }


async def _get_missions_with_positions() -> List[Dict[str, Any]]:
    """Fetch active missions that may have open positions."""
    from app.services.database import get_active_missions
    try:
        return await get_active_missions()
    except Exception as e:
        logger.error("Failed to fetch missions with positions", error=str(e))
        return []


async def _audit_log(**kwargs) -> None:
    """Write an audit log entry, swallowing errors to avoid disrupting the pipeline."""
    try:
        from app.services.database import record_agent_audit
        await record_agent_audit(**kwargs)
    except Exception as e:
        logger.warning("Failed to write audit log", error=str(e))

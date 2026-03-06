"""
Layer B: User Filter Node
Filters signals for eligible missions and generates execution payloads
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List

import structlog

from app.services.circuit_breaker import get_circuit_breaker
from app.workflows.state import AgentState, ExecutionPayload

logger = structlog.get_logger(__name__)


async def user_filter_node(state: AgentState) -> Dict[str, Any]:
    """
    User Filter Node (Layer B).

    Steps:
    1. Fetch active missions from database
    2. Validate each mission's agent approval status
    3. Apply time-bound logic (force close on last days)
    4. Match signals to user risk profiles
    5. Generate execution payloads

    Args:
        state: Current workflow state with signals

    Returns:
        Updated state with execution payloads
    """
    from app.services.hyperliquid import HyperliquidClient, AgentWalletManager
    from app.services.llm import DeepSeekClient

    signals = state.get("signals", [])
    errors = state.get("errors", [])

    if not signals:
        logger.info("No signals to process")
        return {
            "eligible_missions": [],
            "execution_payloads": [],
            "skip_reasons": {},
            "errors": errors,
            "completed_nodes": state.get("completed_nodes", []) + ["user_filter"],
        }

    logger.info("Starting user filter", signals_count=len(signals))

    # Initialize services
    hl_client = HyperliquidClient()
    agent_manager = AgentWalletManager(hl_client)
    llm = DeepSeekClient()

    eligible_missions = []
    execution_payloads: List[ExecutionPayload] = []
    skip_reasons: Dict[str, str] = {}

    try:
        # Fetch active missions from database
        missions = await _get_active_missions()

        if not missions:
            logger.info("No active missions found")
            return {
                "eligible_missions": [],
                "execution_payloads": [],
                "skip_reasons": {},
                "errors": errors,
                "completed_nodes": state.get("completed_nodes", []) + ["user_filter"],
            }

        circuit_breaker = get_circuit_breaker()

        for mission in missions:
            mission_id = mission["id"]
            user_address = mission.get("user_wallet_address")

            logger.debug("Processing mission", mission_id=mission_id)

            # Skip missions with tripped circuit breaker
            if circuit_breaker.is_tripped(mission_id):
                cb_status = circuit_breaker.get_status(mission_id)
                skip_reasons[mission_id] = f"Circuit breaker tripped ({cb_status['failure_count']} failures, resets at {cb_status.get('reset_at', 'unknown')})"
                logger.warning(
                    "Skipping mission due to circuit breaker",
                    mission_id=mission_id,
                    failure_count=cb_status["failure_count"],
                )
                continue

            try:
                # Calculate mission day
                started_at = mission.get("started_at")
                if started_at:
                    if isinstance(started_at, str):
                        started_at = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                    mission_day = (datetime.utcnow() - started_at.replace(tzinfo=None)).days + 1
                else:
                    mission_day = 1

                duration_days = mission.get("duration_days", 30)
                days_remaining = duration_days - mission_day + 1

                # CRITICAL: Force close on last 2 days
                if days_remaining <= 2:
                    # Get user's current positions
                    positions = await hl_client.get_positions(user_address)
                    if positions:
                        # Fetch market prices for slippage-protected limit orders
                        market_data = state.get("market_data", {})

                        for pos in positions:
                            # Calculate limit price with 2% slippage protection
                            asset_market = market_data.get(pos["asset"], {})
                            mid_price = asset_market.get("price")

                            if mid_price and mid_price > 0:
                                if pos["direction"] == "LONG":
                                    # Selling long: accept 2% below mid
                                    limit_price = mid_price * 0.98
                                else:
                                    # Buying back short: accept 2% above mid
                                    limit_price = mid_price * 1.02
                            else:
                                limit_price = None  # Fallback to market order if no price data

                            execution_payloads.append({
                                "mission_id": mission_id,
                                "user_id": mission["user_id"],
                                "action": f"EXIT_{pos['direction']}",
                                "asset": pos["asset"],
                                "quantity": pos["size"],
                                "price": limit_price,
                                "leverage": pos.get("leverage", 1),
                                "stop_loss_price": None,
                                "take_profit_price": None,
                                "signal_id": None,
                                "decision_reason": f"Mission ending (day {mission_day} of {duration_days}). Force closing all positions.",
                                "max_slippage_percent": 2.0,
                            })

                        # Audit: Log force close decision
                        await _audit_log(
                            node="user_filter",
                            action="force_close_scheduled",
                            mission_id=mission_id,
                            user_id=mission.get("user_id"),
                            decision={
                                "positions_count": len(positions),
                                "mission_day": mission_day,
                                "duration_days": duration_days,
                                "days_remaining": days_remaining,
                            },
                            reasoning=f"Mission ending (day {mission_day} of {duration_days}). Force closing {len(positions)} positions.",
                        )

                        logger.info(
                            "Force close payloads created",
                            mission_id=mission_id,
                            positions_count=len(positions),
                        )
                    continue

                # Validate agent approval
                if not mission.get("hyperliquid_approved", False):
                    skip_reasons[mission_id] = "Hyperliquid agent not approved"
                    continue

                # Verify on-chain approval
                approval = await agent_manager.check_user_approval(user_address)
                if not approval.get("approved"):
                    skip_reasons[mission_id] = "Agent approval revoked on-chain"
                    continue

                # Get user's current positions
                positions = await hl_client.get_positions(user_address)
                existing_positions = [
                    {
                        "asset": p["asset"],
                        "direction": p["direction"],
                        "leverage": p.get("leverage", 1),
                        "unrealized_pnl": (p["unrealized_pnl"] / p.get("margin_used", 1) * 100) if p.get("margin_used") else 0,
                    }
                    for p in positions
                ]

                # Get account value
                account = await hl_client.get_account_value(user_address)
                current_value = account.get("account_value", mission.get("initial_capital", 0))

                # Calculate PnL
                initial_capital = float(mission.get("initial_capital", 0))
                if initial_capital > 0:
                    total_pnl_percent = ((current_value - initial_capital) / initial_capital) * 100
                else:
                    total_pnl_percent = 0

                # Prepare mission context for LLM
                mission_context = {
                    "id": mission_id,
                    "risk_level": mission.get("risk_level", "MODERATE"),
                    "max_leverage": mission.get("max_leverage", 2),
                    "allowed_assets": mission.get("allowed_assets", ["ETH-USD", "BTC-USD"]),
                    "mission_day": mission_day,
                    "duration_days": duration_days,
                    "days_remaining": days_remaining,
                    "current_value": current_value,
                    "total_pnl_percent": total_pnl_percent,
                }

                # Process each signal for this mission
                for signal in signals:
                    asset = signal["asset"]

                    # Check if asset is allowed for this mission
                    if asset not in mission_context["allowed_assets"]:
                        continue

                    # Check if already have a position in this asset
                    has_position = any(p["asset"] == asset for p in existing_positions)
                    if has_position:
                        # Skip new entries, but consider exits
                        continue

                    # Use LLM to determine if this trade is appropriate for user
                    filter_result = await llm.filter_for_user(
                        signal=signal,
                        mission=mission_context,
                        existing_positions=existing_positions,
                    )

                    # Application-level audit log
                    logger.info(
                        "AUDIT user_filter LLM decision",
                        mission_id=mission_id,
                        asset=asset,
                        should_execute=filter_result.get("should_execute"),
                        adjusted_leverage=filter_result.get("adjusted_leverage"),
                        position_size_pct=filter_result.get("position_size_percent"),
                        skip_reason=filter_result.get("skip_reason"),
                        tokens_used=filter_result.get("_audit_tokens"),
                    )

                    # DB audit log
                    await _audit_log(
                        node="user_filter",
                        action="llm_call",
                        mission_id=mission_id,
                        user_id=mission.get("user_id"),
                        asset=asset,
                        llm_prompt=filter_result.get("_audit_prompt"),
                        llm_response=filter_result.get("_audit_response"),
                        llm_model=filter_result.get("_audit_model"),
                        llm_tokens=filter_result.get("_audit_tokens"),
                        decision={
                            "should_execute": filter_result.get("should_execute"),
                            "adjusted_leverage": filter_result.get("adjusted_leverage"),
                            "position_size_percent": filter_result.get("position_size_percent"),
                        },
                        reasoning=filter_result.get("skip_reason") or filter_result.get("reasoning", ""),
                        success="_audit_error" not in filter_result,
                        error_message=filter_result.get("_audit_error"),
                    )

                    if filter_result.get("should_execute", False):
                        # Adjust leverage based on user's risk level
                        adjusted_leverage = min(
                            filter_result.get("adjusted_leverage", 1),
                            mission_context["max_leverage"],
                        )

                        # Calculate position size
                        position_size_percent = filter_result.get("position_size_percent", 10)
                        available_margin = account.get("withdrawable", 0)
                        position_margin = available_margin * (position_size_percent / 100)

                        # Get current price
                        market_price = state.get("market_data", {}).get(asset, {}).get("price", 0)
                        if market_price > 0:
                            position_size = (position_margin * adjusted_leverage) / market_price
                        else:
                            position_size = 0

                        if position_size > 0:
                            # Calculate stop loss and take profit prices
                            if signal["direction"] == "LONG":
                                stop_loss = market_price * (1 - signal["stop_loss_percent"] / 100)
                                take_profit = market_price * (1 + signal["take_profit_percent"] / 100)
                            else:
                                stop_loss = market_price * (1 + signal["stop_loss_percent"] / 100)
                                take_profit = market_price * (1 - signal["take_profit_percent"] / 100)

                            payload: ExecutionPayload = {
                                "mission_id": mission_id,
                                "user_id": mission["user_id"],
                                "action": f"ENTER_{signal['direction']}",
                                "asset": asset,
                                "quantity": position_size,
                                "price": market_price,
                                "leverage": adjusted_leverage,
                                "stop_loss_price": stop_loss,
                                "take_profit_price": take_profit,
                                "signal_id": f"sig_{datetime.utcnow().strftime('%Y%m%d_%H%M')}_{asset[:3]}",
                                "decision_reason": f"{signal['strategy_tag']}: {signal['reasoning']}",
                                "_started_at": mission.get("started_at"),
                                "_user_balance": current_value,
                            }
                            execution_payloads.append(payload)

                            # Audit: Log approved filter decision
                            await _audit_log(
                                node="user_filter",
                                action="filter_approved",
                                mission_id=mission_id,
                                user_id=mission.get("user_id"),
                                asset=asset,
                                decision={
                                    "direction": signal["direction"],
                                    "leverage": adjusted_leverage,
                                    "position_size": position_size,
                                    "market_price": market_price,
                                },
                                reasoning=f"{signal.get('strategy_tag', '')}: {signal.get('reasoning', '')}",
                            )

                            logger.info(
                                "Execution payload created",
                                mission_id=mission_id,
                                asset=asset,
                                direction=signal["direction"],
                                leverage=adjusted_leverage,
                                size=position_size,
                            )
                    else:
                        skip_reason = filter_result.get("skip_reason", "Not suitable for user profile")

                        # Audit: Log rejected filter decision
                        await _audit_log(
                            node="user_filter",
                            action="filter_rejected",
                            mission_id=mission_id,
                            user_id=mission.get("user_id"),
                            asset=asset,
                            reasoning=skip_reason,
                        )

                        logger.debug(
                            "Signal skipped for mission",
                            mission_id=mission_id,
                            asset=asset,
                            reason=skip_reason,
                        )

                eligible_missions.append(mission_context)

            except Exception as e:
                logger.error(
                    "Error processing mission",
                    mission_id=mission_id,
                    error=str(e),
                )
                errors.append(f"User filter error for mission {mission_id}: {str(e)}")
                skip_reasons[mission_id] = f"Processing error: {str(e)}"

    finally:
        await hl_client.close()
        await llm.close()

    logger.info(
        "User filter completed",
        eligible_missions=len(eligible_missions),
        execution_payloads=len(execution_payloads),
        skipped=len(skip_reasons),
    )

    return {
        "eligible_missions": eligible_missions,
        "execution_payloads": execution_payloads,
        "skip_reasons": skip_reasons,
        "errors": errors,
        "completed_nodes": state.get("completed_nodes", []) + ["user_filter"],
    }


async def _get_active_missions() -> List[Dict[str, Any]]:
    """Fetch active missions from database."""
    from app.services.database import get_active_missions
    try:
        return await get_active_missions()
    except Exception as e:
        logger.error("Failed to fetch active missions from database", error=str(e))
        return []


async def _audit_log(**kwargs) -> None:
    """Write an audit log entry, swallowing errors to avoid disrupting the pipeline."""
    try:
        from app.services.database import record_agent_audit
        await record_agent_audit(**kwargs)
    except Exception as e:
        logger.warning("Failed to write audit log", error=str(e))

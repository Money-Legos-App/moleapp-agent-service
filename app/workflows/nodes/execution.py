"""
Layer C: Execution Node
Executes trades by requesting signatures from wallet-service and submitting to Hyperliquid
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List

import structlog

from app.config import get_settings
from app.services.circuit_breaker import get_circuit_breaker
from app.workflows.state import AgentState, ExecutionResult

logger = structlog.get_logger(__name__)


async def execution_node(state: AgentState) -> Dict[str, Any]:
    """
    Execution Node (Layer C).

    Steps:
    1. Check DRY_RUN mode - if enabled, log but don't execute
    2. Build order payloads for each execution
    3. Request batch signatures from wallet-service (Turnkey)
    4. Submit signed orders to Hyperliquid
    5. Record execution results

    Args:
        state: Current workflow state with execution payloads

    Returns:
        Updated state with execution results
    """
    from app.services.hyperliquid import HyperliquidClient
    from app.services.wallet import TurnkeyBridge

    settings = get_settings()
    execution_payloads = state.get("execution_payloads", [])
    errors = state.get("errors", [])

    # ==================
    # DRY RUN MODE CHECK
    # ==================
    if settings.dry_run:
        logger.info(
            "DRY RUN MODE: Would execute orders (no actual trades)",
            payloads_count=len(execution_payloads),
        )

        # Log details if enabled
        if settings.dry_run_log_trades and execution_payloads:
            for payload in execution_payloads:
                logger.info(
                    "DRY RUN: Would execute",
                    mission_id=payload.get("mission_id"),
                    asset=payload.get("asset"),
                    action=payload.get("action"),
                    quantity=payload.get("quantity"),
                    price=payload.get("price"),
                    leverage=payload.get("leverage"),
                    reason=payload.get("reason"),
                )

        # Return simulated success for all orders
        simulated_orders = [
            {
                "mission_id": p.get("mission_id"),
                "success": True,
                "order_id": f"dry_run_{i}",
                "tx_hash": None,
                "error": None,
                "executed_price": p.get("price"),
                "executed_quantity": p.get("quantity"),
                "dry_run": True,
            }
            for i, p in enumerate(execution_payloads)
        ]

        # Record simulated trades in database for tracking
        for i, payload in enumerate(execution_payloads):
            await _record_execution(
                mission_id=payload.get("mission_id", ""),
                payload=payload,
                result={"success": True, "tx_hash": None, "order_id": f"dry_run_{i}"},
                dry_run=True,
            )

            # Application-level audit log
            logger.info(
                "AUDIT execution DRY_RUN trade",
                mission_id=payload.get("mission_id"),
                asset=payload.get("asset"),
                action=payload.get("action"),
                quantity=payload.get("quantity"),
                price=payload.get("price"),
                leverage=payload.get("leverage"),
                decision_reason=payload.get("decision_reason", "")[:200],
                dry_run=True,
            )

            # DB audit log
            await _audit_log(
                node="execution",
                action="trade_executed",
                mission_id=payload.get("mission_id"),
                user_id=payload.get("user_id"),
                asset=payload.get("asset"),
                decision={
                    "action": payload.get("action"),
                    "quantity": payload.get("quantity"),
                    "price": payload.get("price"),
                    "leverage": payload.get("leverage"),
                    "order_id": f"dry_run_{i}",
                },
                reasoning=payload.get("decision_reason", ""),
                dry_run=True,
                success=True,
            )

        return {
            "executed_orders": simulated_orders,
            "failed_orders": [],
            "errors": errors,
            "dry_run": True,
            "completed_nodes": state.get("completed_nodes", []) + ["execution"],
        }

    if not execution_payloads:
        logger.info("No execution payloads to process")
        return {
            "executed_orders": [],
            "failed_orders": [],
            "errors": errors,
            "completed_nodes": state.get("completed_nodes", []) + ["execution"],
        }

    logger.info("Starting execution", payloads_count=len(execution_payloads))

    # Initialize services
    hl_client = HyperliquidClient()
    wallet_bridge = TurnkeyBridge()

    executed_orders: List[ExecutionResult] = []
    failed_orders: List[ExecutionResult] = []

    try:
        # Group payloads by mission for batch signing
        payloads_by_mission: Dict[str, List[Dict]] = {}
        for payload in execution_payloads:
            mission_id = payload["mission_id"]
            if mission_id not in payloads_by_mission:
                payloads_by_mission[mission_id] = []
            payloads_by_mission[mission_id].append(payload)

        circuit_breaker = get_circuit_breaker()

        # Process each mission's orders
        for mission_id, payloads in payloads_by_mission.items():
            # Skip missions with tripped circuit breaker
            if circuit_breaker.is_tripped(mission_id):
                logger.warning(
                    "Skipping mission execution — circuit breaker tripped",
                    mission_id=mission_id,
                )
                for payload in payloads:
                    failed_orders.append({
                        "mission_id": mission_id,
                        "success": False,
                        "order_id": None,
                        "tx_hash": None,
                        "error": "Circuit breaker tripped — mission temporarily disabled",
                        "executed_price": None,
                        "executed_quantity": None,
                    })
                continue

            logger.info(
                "Processing mission orders",
                mission_id=mission_id,
                orders_count=len(payloads),
            )

            try:
                # Build order payloads
                order_requests = []
                for payload in payloads:
                    is_buy = payload["action"].startswith("ENTER_LONG") or payload["action"].startswith("EXIT_SHORT")

                    order_payload = hl_client.build_order_payload(
                        asset=payload["asset"],
                        is_buy=is_buy,
                        size=payload["quantity"],
                        price=payload.get("price"),  # None for market orders
                        reduce_only="EXIT" in payload["action"],
                        order_type="market" if payload.get("price") is None else "limit",
                    )

                    order_requests.append({
                        "mission_id": mission_id,
                        "payload": order_payload,
                        "original": payload,
                    })

                # Request batch signatures from wallet-service
                signing_results = await wallet_bridge.batch_sign_trades(
                    orders=[
                        {
                            "mission_id": r["mission_id"],
                            "payload": r["payload"],
                        }
                        for r in order_requests
                    ]
                )

                # Submit signed orders to Hyperliquid
                for i, (request, sign_result) in enumerate(zip(order_requests, signing_results)):
                    original_payload = request["original"]

                    if not sign_result.get("success"):
                        failed_orders.append({
                            "mission_id": mission_id,
                            "success": False,
                            "order_id": None,
                            "tx_hash": None,
                            "error": sign_result.get("error", "Signing failed"),
                            "executed_price": None,
                            "executed_quantity": None,
                        })

                        # Audit: Log signing failure
                        await _audit_log(
                            node="execution",
                            action="trade_failed",
                            mission_id=mission_id,
                            user_id=original_payload.get("user_id"),
                            asset=original_payload["asset"],
                            decision={"action": original_payload["action"], "stage": "signing"},
                            success=False,
                            error_message=sign_result.get("error", "Signing failed"),
                        )

                        logger.error(
                            "Order signing failed",
                            mission_id=mission_id,
                            asset=original_payload["asset"],
                            error=sign_result.get("error"),
                        )
                        continue

                    # Submit to Hyperliquid
                    try:
                        submit_result = await hl_client.place_order(sign_result)

                        if submit_result.get("success"):
                            circuit_breaker.record_success(mission_id)

                            executed_orders.append({
                                "mission_id": mission_id,
                                "success": True,
                                "order_id": submit_result.get("order_id"),
                                "tx_hash": submit_result.get("tx_hash"),
                                "error": None,
                                "executed_price": original_payload.get("price"),
                                "executed_quantity": original_payload.get("quantity"),
                            })

                            logger.info(
                                "AUDIT execution LIVE trade success",
                                mission_id=mission_id,
                                asset=original_payload["asset"],
                                action=original_payload["action"],
                                quantity=original_payload.get("quantity"),
                                price=original_payload.get("price"),
                                leverage=original_payload.get("leverage"),
                                order_id=submit_result.get("order_id"),
                                tx_hash=submit_result.get("tx_hash"),
                                decision_reason=original_payload.get("decision_reason", "")[:200],
                            )

                            # Record execution in database
                            await _record_execution(
                                mission_id=mission_id,
                                payload=original_payload,
                                result=submit_result,
                            )

                            # Audit: Log successful live trade
                            await _audit_log(
                                node="execution",
                                action="trade_executed",
                                mission_id=mission_id,
                                user_id=original_payload.get("user_id"),
                                asset=original_payload["asset"],
                                decision={
                                    "action": original_payload["action"],
                                    "quantity": original_payload.get("quantity"),
                                    "price": original_payload.get("price"),
                                    "leverage": original_payload.get("leverage"),
                                    "order_id": submit_result.get("order_id"),
                                    "tx_hash": submit_result.get("tx_hash"),
                                },
                                reasoning=original_payload.get("decision_reason", ""),
                                dry_run=False,
                                success=True,
                            )
                        else:
                            failed_orders.append({
                                "mission_id": mission_id,
                                "success": False,
                                "order_id": None,
                                "tx_hash": None,
                                "error": submit_result.get("error", "Order submission failed"),
                                "executed_price": None,
                                "executed_quantity": None,
                            })

                            # Audit: Log submission failure
                            await _audit_log(
                                node="execution",
                                action="trade_failed",
                                mission_id=mission_id,
                                user_id=original_payload.get("user_id"),
                                asset=original_payload["asset"],
                                decision={"action": original_payload["action"], "stage": "submission"},
                                success=False,
                                error_message=submit_result.get("error", "Order submission failed"),
                            )

                            logger.error(
                                "Order submission failed",
                                mission_id=mission_id,
                                asset=original_payload["asset"],
                                error=submit_result.get("error"),
                            )

                    except Exception as e:
                        failed_orders.append({
                            "mission_id": mission_id,
                            "success": False,
                            "order_id": None,
                            "tx_hash": None,
                            "error": str(e),
                            "executed_price": None,
                            "executed_quantity": None,
                        })
                        logger.error(
                            "Order submission exception",
                            mission_id=mission_id,
                            error=str(e),
                        )

            except Exception as e:
                circuit_breaker.record_failure(mission_id, str(e))

                logger.error(
                    "Mission execution failed",
                    mission_id=mission_id,
                    error=str(e),
                )
                errors.append(f"Execution error for mission {mission_id}: {str(e)}")

                # Mark all orders for this mission as failed
                for payload in payloads:
                    failed_orders.append({
                        "mission_id": mission_id,
                        "success": False,
                        "order_id": None,
                        "tx_hash": None,
                        "error": str(e),
                        "executed_price": None,
                        "executed_quantity": None,
                    })

    finally:
        await hl_client.close()
        await wallet_bridge.close()

    logger.info(
        "Execution completed",
        executed=len(executed_orders),
        failed=len(failed_orders),
    )

    return {
        "executed_orders": executed_orders,
        "failed_orders": failed_orders,
        "errors": errors,
        "completed_nodes": state.get("completed_nodes", []) + ["execution"],
    }


async def _record_execution(
    mission_id: str,
    payload: Dict[str, Any],
    result: Dict[str, Any],
    dry_run: bool = False,
) -> None:
    """Record a trade execution in the database."""
    from app.services.database import record_trade_execution

    try:
        # Calculate mission_day from mission start
        started_at = payload.get("_started_at")
        if started_at:
            if isinstance(started_at, str):
                started_at = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            mission_day = (datetime.utcnow() - started_at.replace(tzinfo=None)).days + 1
        else:
            mission_day = 1

        await record_trade_execution(
            mission_id=mission_id,
            action=payload.get("action", ""),
            asset=payload.get("asset", ""),
            quantity=payload.get("quantity", 0),
            price=payload.get("price"),
            mission_day=mission_day,
            decision_reason=payload.get("decision_reason", ""),
            user_balance=payload.get("_user_balance", 0),
            success=result.get("success", False),
            tx_hash=result.get("tx_hash"),
            error_message=result.get("error"),
            signal_id=payload.get("signal_id"),
            dry_run=dry_run,
        )

        logger.info(
            "Execution recorded",
            mission_id=mission_id,
            action=payload.get("action"),
            asset=payload.get("asset"),
            success=result.get("success"),
        )
    except Exception as e:
        logger.error("Failed to record execution", error=str(e), mission_id=mission_id)


async def _audit_log(**kwargs) -> None:
    """Write an audit log entry, swallowing errors to avoid disrupting the pipeline."""
    try:
        from app.services.database import record_agent_audit
        await record_agent_audit(**kwargs)
    except Exception as e:
        logger.warning("Failed to write audit log", error=str(e))

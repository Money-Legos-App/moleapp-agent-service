"""
Position Monitoring Tasks

Two-phase monitoring that runs every 5 minutes:
  Phase A: Risk Enforcement — deterministic SL/TP/trailing/drawdown closes
  Phase B: Informational Alerts — liquidation warnings, large loss alerts

Infrastructure Resilience:
  - Primary monitor writes heartbeat to Redis each cycle
  - Backup monitor (60s) checks heartbeat and takes over if missing
"""

import json
import time
from typing import List, Dict, Any

import structlog

from app.config import get_settings

logger = structlog.get_logger(__name__)

HEARTBEAT_KEY = "agent:risk:heartbeat"
MONITOR_UNHEALTHY_KEY = "agent:risk:monitor_unhealthy"


async def check_positions_for_alerts() -> List[Dict[str, Any]]:
    """
    Check all open positions for risk events.

    Phase A: Risk enforcement — evaluate SL/TP/trailing/drawdown and force-close
    Phase B: Informational alerts — liquidation warnings, large losses

    Returns:
        List of alert dictionaries
    """
    from app.services.hyperliquid import HyperliquidClient
    from app.services.execution_queue import get_redis
    from app.services.database import get_open_positions
    from app.services.risk_manager import evaluate_mission_risk, execute_risk_closes
    from app.services.wallet import TurnkeyBridge

    settings = get_settings()

    logger.info("Position monitoring: starting")

    hl_client = HyperliquidClient()
    wallet_bridge = TurnkeyBridge()
    alerts = []
    risk_summary = {"missions_checked": 0, "positions_closed": 0, "kill_switches": 0}

    try:
        # Read cached prices from Redis for mark price reference
        redis = await get_redis()

        # Write heartbeat: "primary risk monitor is alive"
        await redis.set(
            HEARTBEAT_KEY,
            json.dumps({"ts": time.time(), "source": "primary"}),
            ex=settings.heartbeat_ttl_seconds,
        )

        cached_prices = {}
        raw_prices = await redis.hgetall("agent:market:prices")
        for coin, data_str in raw_prices.items():
            try:
                cached_prices[coin] = json.loads(data_str)
            except (json.JSONDecodeError, TypeError):
                pass

        # Get active missions
        missions = await _get_active_missions()

        for mission in missions:
            user_address = mission.get("user_wallet_address")
            if not user_address:
                continue

            mission_id = mission["id"]

            try:
                # Single API call per mission (rate-limited at client level)
                state = await hl_client.get_clearinghouse_state(user_address)
                hl_positions = state["positions"]
                account_value = state.get("account_value", 0)

                if not hl_positions:
                    continue

                risk_summary["missions_checked"] += 1

                # ========================================
                # Phase A: Risk Enforcement
                # ========================================
                if settings.risk_enforcement_enabled:
                    # Load DB positions (with SL/TP prices)
                    db_positions = await get_open_positions(mission_id)

                    actions, kill_switch = await evaluate_mission_risk(
                        mission=mission,
                        hl_positions=hl_positions,
                        account_value=account_value,
                        db_positions=db_positions,
                        cached_prices=cached_prices,
                        redis=redis,
                    )

                    if kill_switch:
                        risk_summary["kill_switches"] += 1

                    if actions:
                        result = await execute_risk_closes(
                            actions=actions,
                            mission=mission,
                            hl_client=hl_client,
                            wallet_bridge=wallet_bridge,
                        )
                        risk_summary["positions_closed"] += result.get("closed", 0)

                        # Generate alerts for risk closes
                        for action in actions:
                            alerts.append({
                                "type": f"RISK_CLOSE_{action.reason}",
                                "urgency": "high",
                                "mission_id": mission_id,
                                "user_id": mission.get("user_id"),
                                "asset": action.asset,
                                "direction": action.direction,
                                "current_price": action.current_price,
                                "trigger_price": action.trigger_price,
                                "message": f"{action.reason}: {action.direction} {action.asset} closed at {action.current_price}",
                            })

                # ========================================
                # Phase B: Informational Alerts
                # ========================================
                for pos in hl_positions:
                    asset = pos["asset"]
                    coin = asset.replace("-USD", "")

                    # Use cached mark price if available
                    if coin in cached_prices:
                        current_price = cached_prices[coin].get("markPx", pos["entry_price"])
                    else:
                        current_price = pos["entry_price"]

                    liquidation_price = pos.get("liquidation_price", 0)

                    # Check liquidation proximity (informational — Phase A handles the close)
                    if liquidation_price > 0:
                        if pos["direction"] == "LONG":
                            distance_pct = ((current_price - liquidation_price) / current_price) * 100
                        else:
                            distance_pct = ((liquidation_price - current_price) / current_price) * 100

                        if distance_pct < 10:
                            alerts.append({
                                "type": "LIQUIDATION_WARNING",
                                "urgency": "high" if distance_pct < 5 else "medium",
                                "mission_id": mission_id,
                                "user_id": mission.get("user_id"),
                                "asset": asset,
                                "direction": pos["direction"],
                                "current_price": current_price,
                                "liquidation_price": liquidation_price,
                                "distance_percent": distance_pct,
                                "message": f"Position {distance_pct:.1f}% away from liquidation",
                            })

                    # Check large unrealized loss
                    entry_price = pos["entry_price"]
                    if entry_price > 0:
                        if pos["direction"] == "LONG":
                            pnl_pct = ((current_price - entry_price) / entry_price) * 100
                        else:
                            pnl_pct = ((entry_price - current_price) / entry_price) * 100

                        if pnl_pct < -15:
                            alerts.append({
                                "type": "LARGE_LOSS_WARNING",
                                "urgency": "medium",
                                "mission_id": mission_id,
                                "user_id": mission.get("user_id"),
                                "asset": asset,
                                "direction": pos["direction"],
                                "unrealized_pnl_percent": pnl_pct,
                                "message": f"Position has {pnl_pct:.1f}% unrealized loss",
                            })

            except Exception as e:
                logger.error(
                    "Error checking mission positions",
                    mission_id=mission_id,
                    error=str(e),
                )
                continue

    finally:
        await hl_client.close()
        await wallet_bridge.close()

    # Log summary
    logger.info(
        "Position monitoring: complete",
        missions_checked=risk_summary["missions_checked"],
        positions_closed=risk_summary["positions_closed"],
        kill_switches=risk_summary["kill_switches"],
        alerts_count=len(alerts),
    )

    if alerts:
        await _send_alert_notifications(alerts)

    # Clear unhealthy flag: primary completed successfully
    try:
        redis = await get_redis()
        await redis.delete(MONITOR_UNHEALTHY_KEY)
    except Exception:
        pass

    return alerts


async def _get_active_missions() -> List[Dict[str, Any]]:
    """Fetch active missions."""
    from app.services.database import get_active_missions
    try:
        return await get_active_missions()
    except Exception as e:
        logger.error("Failed to fetch active missions for monitoring", error=str(e))
        return []


async def _send_alert_notifications(alerts: List[Dict[str, Any]]) -> None:
    """Send notifications for alerts."""
    # TODO: Implement notification sending via notification-service
    for alert in alerts:
        if alert.get("urgency") == "high":
            logger.warning(
                "HIGH URGENCY ALERT",
                type=alert["type"],
                mission_id=alert["mission_id"],
                message=alert.get("message"),
            )


async def backup_risk_monitor() -> Dict[str, Any]:
    """
    Dead-man's-switch backup monitor.

    Runs every 60 seconds via scheduler. If primary heartbeat is missing
    (TTL expired), takes over risk evaluation to prevent positions from
    going unmonitored during agent service issues.

    Also sets a Redis flag that blocks new position entries when the
    primary monitor is unhealthy.
    """
    from app.services.execution_queue import get_redis

    redis = await get_redis()
    settings = get_settings()

    heartbeat_raw = await redis.get(HEARTBEAT_KEY)

    if heartbeat_raw:
        # Primary is alive — nothing to do
        return {"status": "primary_healthy", "action": "none"}

    # Heartbeat missing — primary may be down
    logger.warning("BACKUP MONITOR: Primary heartbeat missing, taking over risk checks")

    # Set unhealthy flag (checked by execution pre-filter to block new entries)
    await redis.set(MONITOR_UNHEALTHY_KEY, "1", ex=settings.heartbeat_ttl_seconds)

    if not settings.risk_enforcement_enabled:
        return {"status": "primary_down", "action": "risk_enforcement_disabled"}

    # Run the same risk evaluation as the primary
    try:
        alerts = await check_positions_for_alerts()
        return {
            "status": "primary_down",
            "action": "backup_executed",
            "alerts": len(alerts),
        }
    except Exception as e:
        logger.error("Backup monitor risk check failed", error=str(e))
        return {"status": "primary_down", "action": "backup_failed", "error": str(e)}

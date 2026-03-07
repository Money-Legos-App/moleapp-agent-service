"""
Fast Actor — Deterministic Execution Engine

Executes Playbooks created by the Slow Thinker (LLM) using live WebSocket
prices for sub-second entries and exits. Zero LLM latency in the hot path.

Architecture:
- Subscribes to WS allMids ticks via callback (in-process, no pub/sub)
- Evaluates pending playbooks for entry zone conditions
- Evaluates active playbooks for SL/TP/trailing stop exits
- Uses IOC orders for exits (immediate fill), GTC limit for entries
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple

import structlog
from redis.asyncio import Redis

from app.config import get_settings
from app.models.playbook import (
    Playbook,
    PlaybookStatus,
    PLAYBOOKS_PENDING_KEY,
    PLAYBOOKS_ACTIVE_KEY,
)
from app.services.risk_manager import (
    check_stop_loss,
    check_take_profit,
    check_trailing_stop,
)

logger = structlog.get_logger(__name__)

FAST_ACTOR_HEALTH_KEY = "agent:fast_actor:healthy"


class FastActor:
    def __init__(self, redis: Redis):
        self._redis = redis
        self._pending: Dict[str, Playbook] = {}    # playbook_id -> Playbook
        self._active: Dict[str, Playbook] = {}     # playbook_id -> Playbook
        self._trailing: Dict[str, float] = {}      # playbook_id -> peak price
        self._running = False
        self._hl_client = None
        self._wallet_bridge = None
        self._executing: set = set()  # playbook IDs currently being executed

    async def start(self):
        """Initialize shared clients and load existing playbooks from Redis."""
        from app.services.hyperliquid import HyperliquidClient
        from app.services.wallet import TurnkeyBridge

        self._hl_client = HyperliquidClient()
        self._wallet_bridge = TurnkeyBridge()
        self._running = True

        # Pre-populate asset index cache (avoid cold-start per order)
        try:
            await self._hl_client._ensure_asset_index_cache()
        except Exception as e:
            logger.warning("Fast Actor: asset index cache warmup failed", error=str(e))

        await self._reload_playbooks()
        asyncio.create_task(self._expiry_loop())

        logger.info(
            "Fast Actor started",
            pending=len(self._pending),
            active=len(self._active),
        )

    async def stop(self):
        """Shutdown and clean up clients."""
        self._running = False
        if self._hl_client:
            await self._hl_client.close()
        if self._wallet_bridge:
            await self._wallet_bridge.close()
        logger.info("Fast Actor stopped")

    async def on_price_tick(self, mids: Dict[str, str]):
        """
        Called on every WS allMids message (~1/sec).
        Must be fast: pure Python checks, no slow awaits in the hot path.
        Orders are dispatched to asyncio tasks for non-blocking execution.
        """
        if not self._running:
            return

        if not self._pending and not self._active:
            return

        orders_to_execute: List[Tuple[str, Playbook, float]] = []

        # Check PENDING playbooks for entry conditions
        expired_ids = []
        for pb_id, pb in list(self._pending.items()):
            if pb_id in self._executing:
                continue

            if pb.is_expired:
                expired_ids.append(pb_id)
                continue

            coin = pb.asset.replace("-USD", "")
            mid_str = mids.get(coin)
            if not mid_str:
                continue

            price = float(mid_str)

            if pb.entry_zone_min <= price <= pb.entry_zone_max:
                slippage = abs(price - pb.entry_price) / pb.entry_price * 100
                if slippage <= pb.max_slippage_pct:
                    orders_to_execute.append(("ENTER", pb, price))
                    self._executing.add(pb_id)

        # Check ACTIVE playbooks for exit conditions
        for pb_id, pb in list(self._active.items()):
            if pb_id in self._executing:
                continue

            coin = pb.asset.replace("-USD", "")
            mid_str = mids.get(coin)
            if not mid_str:
                continue

            price = float(mid_str)

            # Update trailing peak
            if pb.direction == "LONG":
                self._trailing[pb_id] = max(self._trailing.get(pb_id, price), price)
            else:
                self._trailing[pb_id] = min(self._trailing.get(pb_id, price), price)

            # Check stop loss (highest priority)
            if check_stop_loss(pb.direction, price, pb.stop_loss_price):
                orders_to_execute.append(("EXIT_SL", pb, price))
                self._executing.add(pb_id)
                continue

            # Check take profit
            if check_take_profit(pb.direction, price, pb.take_profit_price):
                orders_to_execute.append(("EXIT_TP", pb, price))
                self._executing.add(pb_id)
                continue

            # Check trailing stop
            peak = self._trailing.get(pb_id, price)
            triggered, _ = check_trailing_stop(
                pb.direction, price,
                pb.entered_price or pb.entry_price,
                peak,
                pb.trailing_activation_pct,
                pb.trailing_callback_pct,
            )
            if triggered:
                orders_to_execute.append(("EXIT_TRAILING", pb, price))
                self._executing.add(pb_id)

        # Clean up expired
        for pb_id in expired_ids:
            self._pending.pop(pb_id, None)
            asyncio.create_task(self._update_status(pb_id, PlaybookStatus.EXPIRED))

        # Execute orders asynchronously
        if orders_to_execute:
            asyncio.create_task(self._execute_batch(orders_to_execute))

        # Update health sentinel
        await self._redis.set(FAST_ACTOR_HEALTH_KEY, str(time.time()), ex=60)

    async def _execute_batch(self, orders: List[Tuple[str, Playbook, float]]):
        """Execute a batch of entry/exit orders."""
        for action_type, pb, price in orders:
            try:
                if action_type == "ENTER":
                    await self._execute_entry(pb, price)
                else:
                    await self._execute_exit(pb, price, action_type)
            except Exception as e:
                logger.error(
                    "Fast Actor execution failed",
                    playbook_id=pb.playbook_id,
                    action=action_type,
                    error=str(e),
                )
                # Record failure in circuit breaker
                from app.services.circuit_breaker import get_circuit_breaker
                cb = get_circuit_breaker()
                await cb.record_failure(pb.mission_id, str(e))
            finally:
                self._executing.discard(pb.playbook_id)

    async def _execute_entry(self, pb: Playbook, live_price: float):
        """Execute entry using GTC limit order at live price."""
        # Check circuit breaker
        from app.services.circuit_breaker import get_circuit_breaker
        cb = get_circuit_breaker()
        if await cb.is_tripped(pb.mission_id):
            logger.warning(
                "Fast Actor: circuit breaker tripped, cancelling playbook",
                playbook_id=pb.playbook_id,
                mission_id=pb.mission_id,
            )
            self._pending.pop(pb.playbook_id, None)
            await self._update_status(pb.playbook_id, PlaybookStatus.CANCELLED)
            return

        settings = get_settings()

        # Build and submit order (GTC limit for entries)
        typed_data = self._hl_client.build_eip712_order(
            asset=pb.asset,
            is_buy=(pb.direction == "LONG"),
            size=pb.position_size,
            price=live_price,
            reduce_only=False,
            order_type="limit",
        )

        sign_result = await self._wallet_bridge.sign_with_agent_key(
            mission_id=pb.mission_id,
            typed_data=typed_data,
        )

        if not sign_result.get("success"):
            logger.error(
                "Fast Actor: entry signing failed",
                playbook_id=pb.playbook_id,
                error=sign_result.get("error"),
            )
            return

        if settings.dry_run:
            logger.info(
                "DRY RUN: Fast Actor would enter",
                playbook_id=pb.playbook_id,
                asset=pb.asset,
                direction=pb.direction,
                price=live_price,
            )
            result = {"success": True, "order_id": f"dry_{pb.playbook_id[:8]}"}
        else:
            result = await self._hl_client.place_order_eip712(
                typed_data=typed_data,
                signature=sign_result["signature"],
            )

        if result.get("success"):
            # Persist position to DB
            from app.services.database import create_position, set_position_risk_levels
            from app.services.risk_manager import set_trailing_state

            position_id = await create_position(
                mission_id=pb.mission_id,
                asset=pb.asset,
                direction=pb.direction,
                entry_price=live_price,
                quantity=pb.position_size,
                leverage=pb.leverage,
                margin_used=pb.margin_allocated,
                hyperliquid_order_id=result.get("order_id"),
            )

            await set_position_risk_levels(
                position_id=position_id,
                stop_loss_price=pb.stop_loss_price,
                take_profit_price=pb.take_profit_price,
            )

            await set_trailing_state(
                redis=self._redis,
                position_id=position_id,
                highest_price=live_price,
                lowest_price=live_price,
            )

            # Update playbook state
            pb.status = PlaybookStatus.ENTERED
            pb.entered_at = time.time()
            pb.entered_price = live_price
            pb.position_id = position_id

            self._pending.pop(pb.playbook_id, None)
            self._active[pb.playbook_id] = pb
            self._trailing[pb.playbook_id] = live_price

            await self._save_playbook(pb)

            # Record success in circuit breaker
            await cb.record_success(pb.mission_id)

            logger.info(
                "Fast Actor: ENTERED",
                playbook_id=pb.playbook_id,
                mission_id=pb.mission_id,
                asset=pb.asset,
                direction=pb.direction,
                price=live_price,
                position_id=position_id,
            )
        else:
            logger.error(
                "Fast Actor: entry order failed",
                playbook_id=pb.playbook_id,
                error=result.get("error"),
            )
            await cb.record_failure(pb.mission_id, result.get("error", "order_failed"))

    async def _execute_exit(self, pb: Playbook, live_price: float, reason: str):
        """Execute exit using IOC order for immediate fill."""
        settings = get_settings()

        # Determine order type based on config
        order_type = "market" if settings.fast_actor_use_ioc_exits else "limit"

        typed_data = self._hl_client.build_eip712_order(
            asset=pb.asset,
            is_buy=(pb.direction != "LONG"),  # reverse direction
            size=pb.position_size,
            price=live_price,
            reduce_only=True,
            order_type=order_type,
        )

        sign_result = await self._wallet_bridge.sign_with_agent_key(
            mission_id=pb.mission_id,
            typed_data=typed_data,
        )

        if not sign_result.get("success"):
            logger.error(
                "Fast Actor: exit signing failed",
                playbook_id=pb.playbook_id,
                reason=reason,
                error=sign_result.get("error"),
            )
            return

        if settings.dry_run:
            logger.info(
                "DRY RUN: Fast Actor would exit",
                playbook_id=pb.playbook_id,
                asset=pb.asset,
                reason=reason,
                price=live_price,
            )
            result = {"success": True}
        else:
            result = await self._hl_client.place_order_eip712(
                typed_data=typed_data,
                signature=sign_result["signature"],
            )

        if result.get("success"):
            # Calculate realized PnL
            if pb.entered_price and pb.entered_price > 0:
                if pb.direction == "LONG":
                    pnl = (live_price - pb.entered_price) * pb.position_size
                else:
                    pnl = (pb.entered_price - live_price) * pb.position_size
            else:
                pnl = 0

            # Close position in DB
            if pb.position_id:
                from app.services.database import close_position
                close_reason_map = {
                    "EXIT_SL": "STOP_LOSS",
                    "EXIT_TP": "TAKE_PROFIT",
                    "EXIT_TRAILING": "TRAILING_STOP",
                }
                await close_position(
                    position_id=pb.position_id,
                    close_price=live_price,
                    realized_pnl=pnl,
                    close_reason=close_reason_map.get(reason, "SIGNAL"),
                )

            # Update playbook state
            pb.status = PlaybookStatus.COMPLETED
            self._active.pop(pb.playbook_id, None)
            self._trailing.pop(pb.playbook_id, None)
            await self._save_playbook(pb)

            logger.info(
                "Fast Actor: EXITED",
                playbook_id=pb.playbook_id,
                mission_id=pb.mission_id,
                asset=pb.asset,
                reason=reason,
                entry_price=pb.entered_price,
                exit_price=live_price,
                pnl=round(pnl, 2),
            )
        else:
            logger.error(
                "Fast Actor: exit order failed",
                playbook_id=pb.playbook_id,
                reason=reason,
                error=result.get("error"),
            )

    async def _reload_playbooks(self):
        """Load existing playbooks from Redis on startup."""
        pending_ids = await self._redis.smembers(PLAYBOOKS_PENDING_KEY)
        active_ids = await self._redis.smembers(PLAYBOOKS_ACTIVE_KEY)

        for pb_id in pending_ids:
            pb_id = pb_id if isinstance(pb_id, str) else pb_id.decode()
            raw = await self._redis.get(f"agent:playbook:{pb_id}")
            if raw:
                try:
                    pb = Playbook.from_json(raw if isinstance(raw, str) else raw.decode())
                    if not pb.is_expired:
                        self._pending[pb.playbook_id] = pb
                except Exception:
                    pass

        for pb_id in active_ids:
            pb_id = pb_id if isinstance(pb_id, str) else pb_id.decode()
            raw = await self._redis.get(f"agent:playbook:{pb_id}")
            if raw:
                try:
                    pb = Playbook.from_json(raw if isinstance(raw, str) else raw.decode())
                    self._active[pb.playbook_id] = pb
                    self._trailing[pb.playbook_id] = pb.entered_price or pb.entry_price
                except Exception:
                    pass

        logger.info(
            "Fast Actor: playbooks loaded from Redis",
            pending=len(self._pending),
            active=len(self._active),
        )

    async def _save_playbook(self, pb: Playbook):
        """Save playbook state to Redis and update set membership."""
        pipe = self._redis.pipeline()
        pipe.set(pb.redis_key, pb.to_json(), ex=pb.ttl_seconds + 3600)

        if pb.status == PlaybookStatus.ENTERED:
            pipe.srem(PLAYBOOKS_PENDING_KEY, pb.playbook_id)
            pipe.sadd(PLAYBOOKS_ACTIVE_KEY, pb.playbook_id)
        elif pb.status in (PlaybookStatus.COMPLETED, PlaybookStatus.EXPIRED, PlaybookStatus.CANCELLED):
            pipe.srem(PLAYBOOKS_PENDING_KEY, pb.playbook_id)
            pipe.srem(PLAYBOOKS_ACTIVE_KEY, pb.playbook_id)

        await pipe.execute()

    async def _update_status(self, playbook_id: str, status: PlaybookStatus):
        """Update a playbook's status in Redis."""
        raw = await self._redis.get(f"agent:playbook:{playbook_id}")
        if raw:
            try:
                pb = Playbook.from_json(raw if isinstance(raw, str) else raw.decode())
                pb.status = status
                await self._save_playbook(pb)
            except Exception:
                pass

        # Clean up sets
        pipe = self._redis.pipeline()
        pipe.srem(PLAYBOOKS_PENDING_KEY, playbook_id)
        pipe.srem(PLAYBOOKS_ACTIVE_KEY, playbook_id)
        await pipe.execute()

    async def _expiry_loop(self):
        """Periodically clean up expired playbooks."""
        while self._running:
            try:
                await asyncio.sleep(30)
                expired = [
                    pb_id for pb_id, pb in self._pending.items()
                    if pb.is_expired
                ]
                for pb_id in expired:
                    self._pending.pop(pb_id, None)
                    await self._update_status(pb_id, PlaybookStatus.EXPIRED)

                if expired:
                    logger.info("Fast Actor: expired playbooks cleaned", count=len(expired))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Fast Actor expiry loop error", error=str(e))

    def get_status(self) -> Dict[str, Any]:
        """Get Fast Actor status for health checks."""
        return {
            "running": self._running,
            "pending_playbooks": len(self._pending),
            "active_playbooks": len(self._active),
            "executing": len(self._executing),
        }

"""
Execution Queue (The Hands)

Redis-backed job queue that fans out trade execution to N missions.

Architecture:
- Dispatcher: Takes a MarketState and creates one job per active mission
- Worker: Processes individual mission jobs (filter → sign → submit)
- Rate Limiter: Token bucket for Hyperliquid API calls

Uses arq (async Redis queue) for job management with:
- Automatic retries with backoff
- Concurrency limits
- Job timeouts
- Dead letter handling
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog
from redis.asyncio import Redis

from app.config import get_settings
from app.services.signal_provider import MarketState

logger = structlog.get_logger(__name__)

# Redis key prefixes
QUEUE_KEY = "agent:execution:queue"
PROCESSING_KEY = "agent:execution:processing"
RESULTS_KEY = "agent:execution:results"
MARKET_STATE_KEY = "agent:market_state:latest"
RATE_LIMIT_KEY = "agent:ratelimit:hl"
CYCLE_STATS_KEY = "agent:cycle:stats"


class RateLimiter:
    """
    Token bucket rate limiter backed by Redis.
    Ensures Hyperliquid API calls stay under the rate limit across all workers.
    """

    def __init__(self, redis: Redis, max_per_second: float = 5.0):
        self._redis = redis
        self._max_per_second = max_per_second
        self._interval = 1.0 / max_per_second

    async def acquire(self, timeout: float = 10.0) -> bool:
        """
        Wait for a rate limit token. Returns True if acquired, False if timed out.
        Uses Redis SETNX + TTL for distributed rate limiting.
        """
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            now = time.time()
            # Use a sliding window: store last N timestamps
            pipe = self._redis.pipeline()
            # Remove old entries (older than 1 second)
            pipe.zremrangebyscore(RATE_LIMIT_KEY, 0, now - 1.0)
            # Count entries in the last second
            pipe.zcard(RATE_LIMIT_KEY)
            results = await pipe.execute()
            current_count = results[1]

            if current_count < self._max_per_second:
                # Add our timestamp
                await self._redis.zadd(RATE_LIMIT_KEY, {str(uuid.uuid4()): now})
                await self._redis.expire(RATE_LIMIT_KEY, 5)
                return True

            # Wait a bit before retrying
            await asyncio.sleep(self._interval)

        logger.warning("Rate limit acquisition timed out", timeout=timeout)
        return False


class ExecutionDispatcher:
    """
    Dispatches execution jobs from a MarketState to the Redis queue.
    One job per active mission.
    """

    def __init__(self, redis: Redis):
        self._redis = redis

    async def dispatch(self, market_state: MarketState) -> Dict[str, Any]:
        """
        Create execution jobs for all active missions.

        Steps:
        1. Store MarketState in Redis (read by workers)
        2. Fetch active missions from DB
        3. Create one job per mission and push to queue
        4. Return dispatch stats

        Args:
            market_state: The signals produced by Signal Provider

        Returns:
            Dict with dispatch stats (jobs_created, cycle_id, etc.)
        """
        from app.services.database import get_active_missions

        cycle_id = market_state.cycle_id

        # Store market state in Redis for workers to read
        await self._redis.set(
            MARKET_STATE_KEY,
            market_state.to_json(),
            ex=600,  # 10 minute TTL
        )

        # Fetch active missions
        missions = await get_active_missions()

        if not missions:
            logger.info("No active missions to dispatch", cycle_id=cycle_id)
            return {"cycle_id": cycle_id, "jobs_created": 0, "missions_found": 0}

        if not market_state.signals:
            logger.info(
                "No signals to process, skipping dispatch",
                cycle_id=cycle_id,
                missions=len(missions),
            )
            return {
                "cycle_id": cycle_id,
                "jobs_created": 0,
                "missions_found": len(missions),
                "reason": "no_signals",
            }

        # Create one job per mission
        jobs_created = 0
        pipe = self._redis.pipeline()

        for mission in missions:
            job = {
                "job_id": str(uuid.uuid4()),
                "cycle_id": cycle_id,
                "mission_id": mission["id"],
                "user_id": mission["user_id"],
                "wallet_id": mission["wallet_id"],
                "user_wallet_address": mission.get("user_wallet_address"),
                "risk_level": mission.get("risk_level", "MODERATE"),
                "max_leverage": mission.get("max_leverage", 2),
                "allowed_assets": mission.get("allowed_assets", ["ETH-USD", "BTC-USD"]),
                "initial_capital": mission.get("initial_capital", 0),
                "duration_days": mission.get("duration_days", 30),
                "started_at": mission.get("started_at").isoformat() if mission.get("started_at") else None,
                "ends_at": mission.get("ends_at").isoformat() if mission.get("ends_at") else None,
                "hyperliquid_approved": mission.get("hyperliquid_approved", False),
                "created_at": datetime.utcnow().isoformat(),
                "attempts": 0,
            }
            pipe.rpush(QUEUE_KEY, json.dumps(job, default=str))
            jobs_created += 1

        await pipe.execute()

        # Store cycle stats
        await self._redis.hset(
            f"{CYCLE_STATS_KEY}:{cycle_id}",
            mapping={
                "cycle_id": cycle_id,
                "dispatched_at": datetime.utcnow().isoformat(),
                "jobs_created": str(jobs_created),
                "signals_count": str(len(market_state.signals)),
                "missions_count": str(len(missions)),
                "status": "dispatched",
            },
        )
        await self._redis.expire(f"{CYCLE_STATS_KEY}:{cycle_id}", 3600)

        logger.info(
            "Execution jobs dispatched",
            cycle_id=cycle_id,
            jobs_created=jobs_created,
            signals=len(market_state.signals),
        )

        return {
            "cycle_id": cycle_id,
            "jobs_created": jobs_created,
            "missions_found": len(missions),
            "signals_count": len(market_state.signals),
        }


class ExecutionWorkerPool:
    """
    Pool of async workers that process execution jobs from Redis queue.
    Each worker: dequeue job → apply user filter → sign with agent key → submit to HL.
    """

    def __init__(self, redis: Redis, concurrency: int = 10):
        self._redis = redis
        self._concurrency = concurrency
        self._rate_limiter = RateLimiter(redis, get_settings().hl_rate_limit_per_second)
        self._running = False
        self._tasks: List[asyncio.Task] = []

    async def start(self) -> None:
        """Start the worker pool."""
        if self._running:
            return

        self._running = True
        logger.info("Starting execution worker pool", concurrency=self._concurrency)

        for i in range(self._concurrency):
            task = asyncio.create_task(self._worker_loop(worker_id=i))
            self._tasks.append(task)

    async def stop(self) -> None:
        """Gracefully stop all workers."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("Execution worker pool stopped")

    async def process_cycle(self, timeout: float = 120.0) -> Dict[str, Any]:
        """
        Process all jobs in the current queue concurrently.
        Drains the queue, then runs all jobs in parallel (up to concurrency limit).

        Args:
            timeout: Max time to wait for all jobs

        Returns:
            Cycle results summary
        """
        start_time = time.monotonic()

        # Drain queue into a list
        jobs = []
        while True:
            raw_job = await self._redis.lpop(QUEUE_KEY)
            if raw_job is None:
                break
            jobs.append(json.loads(raw_job))

        if not jobs:
            return {"executed": 0, "failed": 0, "skipped": 0, "duration_seconds": 0.0}

        # Process all jobs concurrently with semaphore for concurrency limit
        sem = asyncio.Semaphore(self._concurrency)

        async def _run(job):
            async with sem:
                return await self._process_job(job)

        results = await asyncio.gather(
            *[_run(job) for job in jobs],
            return_exceptions=True,
        )

        executed = 0
        failed = 0
        skipped = 0
        for result in results:
            if isinstance(result, Exception):
                failed += 1
            elif result.get("skipped"):
                skipped += 1
            elif result.get("success"):
                executed += result.get("orders_executed", 0)
            else:
                failed += 1

        duration = time.monotonic() - start_time

        return {
            "executed": executed,
            "failed": failed,
            "skipped": skipped,
            "duration_seconds": round(duration, 2),
        }

    async def _worker_loop(self, worker_id: int) -> None:
        """Main loop for a single worker."""
        logger.debug("Worker started", worker_id=worker_id)

        while self._running:
            try:
                # Block-pop from queue with 5s timeout
                result = await self._redis.blpop(QUEUE_KEY, timeout=5)
                if result is None:
                    continue  # Timeout, check if still running

                _, raw_job = result
                job = json.loads(raw_job)

                logger.debug(
                    "Worker picked up job",
                    worker_id=worker_id,
                    mission_id=job["mission_id"],
                    cycle_id=job["cycle_id"],
                )

                await self._process_job(job)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Worker error", worker_id=worker_id, error=str(e))
                await asyncio.sleep(1)  # Backoff on error

        logger.debug("Worker stopped", worker_id=worker_id)

    async def _process_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single mission execution job.

        Steps:
        0. Check circuit breaker for this mission
        1. Load MarketState from Redis
        2. Apply user filter logic for this mission
        3. For each matching signal: sign with agent key → submit to HL
        4. Record results

        Args:
            job: The mission execution job

        Returns:
            Job result dict
        """
        from app.services.hyperliquid import HyperliquidClient
        from app.services.wallet import TurnkeyBridge
        from app.services.llm import DeepSeekClient
        from app.services.circuit_breaker import get_circuit_breaker
        from app.services.observability.langfuse_client import get_langfuse
        from app.services.observability.prompt_manager import get_prompt_manager

        settings = get_settings()
        mission_id = job["mission_id"]
        cycle_id = job["cycle_id"]
        user_id = job.get("user_id")
        circuit_breaker = get_circuit_breaker()

        # Langfuse: per-mission trace linked to the cycle session
        lf = get_langfuse()
        pm = get_prompt_manager()
        trace = lf.start_trace(
            name="trading-cycle-filter",
            session_id=cycle_id,
            user_id=user_id,
            tags=[
                "user-filter",
                job.get("risk_level", "MODERATE"),
                settings.environment,
            ],
            metadata={
                "mission_id": mission_id,
                "cycle_id": cycle_id,
                "risk_level": job.get("risk_level"),
            },
        )

        # Check circuit breaker before doing any work (async Redis-backed)
        if await circuit_breaker.is_tripped(mission_id):
            logger.info(
                "Circuit breaker tripped, skipping mission",
                mission_id=mission_id,
                cycle_id=cycle_id,
            )
            return {"success": True, "skipped": True, "reason": "circuit_breaker_tripped"}

        try:
            # Load market state
            raw_state = await self._redis.get(MARKET_STATE_KEY)
            if raw_state is None:
                logger.warning("No market state found", cycle_id=cycle_id)
                return {"success": False, "skipped": True, "reason": "no_market_state"}

            market_state = MarketState.from_json(raw_state)

            if not market_state.signals:
                return {"success": True, "skipped": True, "reason": "no_signals"}

            # Validate mission eligibility
            if not job.get("hyperliquid_approved", False):
                return {"success": True, "skipped": True, "reason": "not_approved"}

            # Calculate mission timing
            started_at = job.get("started_at")
            if started_at:
                started_dt = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                mission_day = (datetime.utcnow() - started_dt.replace(tzinfo=None)).days + 1
            else:
                mission_day = 1

            duration_days = job.get("duration_days", 30)
            days_remaining = duration_days - mission_day + 1

            # Force close on last 2 days - handled separately by monitoring
            if days_remaining <= 2:
                return {"success": True, "skipped": True, "reason": "mission_ending"}

            user_address = job.get("user_wallet_address")
            if not user_address:
                return {"success": False, "skipped": True, "reason": "no_wallet_address"}

            # Initialize services
            hl_client = HyperliquidClient()
            wallet_bridge = TurnkeyBridge()
            llm = DeepSeekClient()

            try:
                # Single API call: positions + account value combined
                state = await hl_client.get_clearinghouse_state(user_address)
                positions = state["positions"]
                account = state["account"]

                existing_positions = [
                    {
                        "asset": p["asset"],
                        "direction": p["direction"],
                        "leverage": p.get("leverage", 1),
                        "margin_used": p.get("margin_used", 0),
                        "unrealized_pnl": (
                            (p["unrealized_pnl"] / p.get("margin_used", 1) * 100)
                            if p.get("margin_used")
                            else 0
                        ),
                    }
                    for p in positions
                ]

                current_value = account.get("account_value", job.get("initial_capital", 0))

                initial_capital = float(job.get("initial_capital", 0))
                total_pnl_percent = (
                    ((current_value - initial_capital) / initial_capital * 100)
                    if initial_capital > 0
                    else 0
                )

                mission_context = {
                    "id": mission_id,
                    "risk_level": job.get("risk_level", "MODERATE"),
                    "max_leverage": job.get("max_leverage", 2),
                    "allowed_assets": job.get("allowed_assets", ["ETH-USD", "BTC-USD"]),
                    "mission_day": mission_day,
                    "duration_days": duration_days,
                    "days_remaining": days_remaining,
                    "current_value": current_value,
                    "total_pnl_percent": total_pnl_percent,
                }

                orders_executed = 0
                orders_failed = 0

                # Block new entries if risk monitor is unhealthy
                if settings.heartbeat_block_new_entries:
                    monitor_unhealthy = await self._redis.get("agent:risk:monitor_unhealthy")
                    if monitor_unhealthy:
                        logger.warning(
                            "Risk monitor unhealthy, blocking new entries",
                            mission_id=mission_id,
                        )
                        return {"success": True, "skipped": True, "reason": "risk_monitor_unhealthy"}

                for signal in market_state.signals:
                    asset = signal["asset"]

                    # Asset allowed check
                    if asset not in mission_context["allowed_assets"]:
                        continue

                    # Already has position in this asset
                    if any(p["asset"] == asset for p in existing_positions):
                        continue

                    # ── Rule-based pre-filter (skip LLM for obvious rejections) ──
                    pre_filter = self._pre_filter_signal(
                        signal=signal,
                        mission_context=mission_context,
                        existing_positions=existing_positions,
                        account=account,
                    )
                    if pre_filter is not None:
                        logger.info(
                            "Signal rejected by pre-filter (LLM skipped)",
                            mission_id=mission_id,
                            asset=asset,
                            reason=pre_filter,
                        )
                        # Langfuse: record pre-filter rejection
                        lf.log_event(
                            trace,
                            name="pre-filter-rejected",
                            input={"asset": asset, "direction": signal.get("direction")},
                            output={"reason": pre_filter},
                        )
                        continue

                    # Fetch user filter prompt from Langfuse (fallback to local)
                    lf_filter_prompt, _ = pm.get_user_filter_prompt(
                        signal=signal,
                        mission=mission_context,
                        existing_positions=existing_positions,
                    )

                    # LLM filter for this user profile (with Langfuse trace)
                    filter_result = await llm.filter_for_user(
                        signal=signal,
                        mission=mission_context,
                        existing_positions=existing_positions,
                        trace=trace,
                        lf_prompt=lf_filter_prompt,
                    )

                    if not filter_result.get("should_execute", False):
                        continue

                    # Calculate position parameters
                    adjusted_leverage = min(
                        filter_result.get("adjusted_leverage", 1),
                        mission_context["max_leverage"],
                    )
                    position_size_percent = filter_result.get("position_size_percent", 10)
                    available_margin = account.get("withdrawable", 0)
                    position_margin = available_margin * (position_size_percent / 100)
                    market_price = market_state.market_data.get(asset, {}).get("price", 0)

                    if market_price <= 0:
                        continue

                    position_size = (position_margin * adjusted_leverage) / market_price

                    # Slippage-aware sizing: reduce in stressed markets
                    if settings.slippage_sizing_enabled and position_size > 0:
                        from app.services.risk_manager import calculate_slippage_adjusted_size
                        coin = asset.replace("-USD", "")
                        redis_coin_raw = await self._redis.hget("agent:market:prices", coin)
                        slippage_price_data = {}
                        if redis_coin_raw:
                            try:
                                slippage_price_data = json.loads(redis_coin_raw)
                            except (json.JSONDecodeError, TypeError):
                                pass
                        position_size, slippage_reduction = calculate_slippage_adjusted_size(
                            base_position_size=position_size,
                            market_price=market_price,
                            cached_price_data=slippage_price_data,
                            max_slippage_reduction_pct=settings.slippage_max_reduction_pct,
                        )
                        if slippage_reduction > 0:
                            logger.info(
                                "Position size reduced for slippage",
                                mission_id=mission_id,
                                asset=asset,
                                reduction_pct=slippage_reduction,
                            )

                    if position_size <= 0:
                        continue

                    # Calculate SL/TP from user's risk profile (with dynamic SL scaling)
                    from app.services.risk_manager import get_risk_profile, calculate_dynamic_stop_loss_pct
                    risk_profile = get_risk_profile(mission_context.get("risk_level", "MODERATE"))
                    tp_pct = risk_profile["take_profit_pct"]

                    # Dynamic SL: at high leverage, tightens SL to stay ahead of liquidation
                    if settings.dynamic_sl_enabled:
                        sl_pct = calculate_dynamic_stop_loss_pct(
                            asset=asset,
                            leverage=adjusted_leverage,
                            profile_sl_pct=risk_profile["stop_loss_pct"],
                            liquidation_buffer_pct=settings.dynamic_sl_buffer_pct,
                        )
                    else:
                        sl_pct = risk_profile["stop_loss_pct"]

                    if signal["direction"] == "LONG":
                        stop_loss = market_price * (1 - sl_pct / 100)
                        take_profit = market_price * (1 + tp_pct / 100)
                    else:
                        stop_loss = market_price * (1 + sl_pct / 100)
                        take_profit = market_price * (1 - tp_pct / 100)

                    # Rate limiting is handled at the client level (HyperliquidRateLimiter)
                    # Execute trade
                    if settings.dry_run:
                        # DRY RUN: log but don't submit
                        logger.info(
                            "DRY RUN: Would execute trade",
                            mission_id=mission_id,
                            asset=asset,
                            direction=signal["direction"],
                            size=position_size,
                            leverage=adjusted_leverage,
                        )
                        order_result = {
                            "success": True,
                            "tx_hash": None,
                            "order_id": f"dry_run_{uuid.uuid4().hex[:8]}",
                        }
                    else:
                        # LIVE: Build EIP-712, sign with agent key, submit
                        order_result = await self._execute_trade(
                            hl_client=hl_client,
                            wallet_bridge=wallet_bridge,
                            mission_id=mission_id,
                            asset=asset,
                            is_buy=signal["direction"] == "LONG",
                            size=position_size,
                            price=market_price,
                            reduce_only=False,
                        )

                    # Record execution
                    await self._record_trade(
                        mission_id=mission_id,
                        signal=signal,
                        market_price=market_price,
                        position_size=position_size,
                        leverage=adjusted_leverage,
                        mission_day=mission_day,
                        current_value=current_value,
                        result=order_result,
                        started_at=started_at,
                    )

                    if order_result.get("success"):
                        orders_executed += 1

                        # Persist position + risk levels to DB
                        try:
                            from app.services.database import create_position, set_position_risk_levels
                            from app.services.risk_manager import set_trailing_state

                            position_id = await create_position(
                                mission_id=mission_id,
                                asset=asset,
                                direction=signal["direction"],
                                entry_price=market_price,
                                quantity=position_size,
                                leverage=adjusted_leverage,
                                margin_used=position_margin,
                                signal_id=signal.get("signal_id"),
                                hyperliquid_order_id=order_result.get("order_id"),
                            )

                            await set_position_risk_levels(
                                position_id=position_id,
                                stop_loss_price=stop_loss,
                                take_profit_price=take_profit,
                            )

                            # Initialize trailing stop state in Redis
                            await set_trailing_state(
                                redis=self._redis,
                                position_id=position_id,
                                highest_price=market_price,
                                lowest_price=market_price,
                            )

                            logger.info(
                                "Position persisted with risk levels",
                                position_id=position_id,
                                mission_id=mission_id,
                                asset=asset,
                                stop_loss=stop_loss,
                                take_profit=take_profit,
                                risk_level=mission_context.get("risk_level"),
                            )
                        except Exception as persist_err:
                            logger.warning(
                                "Failed to persist position",
                                mission_id=mission_id,
                                asset=asset,
                                error=str(persist_err),
                            )
                    else:
                        orders_failed += 1

                result = {
                    "success": True,
                    "mission_id": mission_id,
                    "orders_executed": orders_executed,
                    "orders_failed": orders_failed,
                }

                # Langfuse: update trace with execution outcome
                trace.update(
                    output={
                        "orders_executed": orders_executed,
                        "orders_failed": orders_failed,
                        "mission_id": mission_id,
                        "cycle_id": cycle_id,
                        "dry_run": settings.dry_run,
                    },
                )
                trace.end()

                if orders_executed > 0:
                    await circuit_breaker.record_success(mission_id)
                return result

            finally:
                await hl_client.close()
                await wallet_bridge.close()
                await llm.close()

        except Exception as e:
            logger.error(
                "Job processing failed",
                mission_id=mission_id,
                cycle_id=cycle_id,
                error=str(e),
                exc_info=True,
            )
            await circuit_breaker.record_failure(mission_id, str(e))
            return {"success": False, "error": str(e)}

    @staticmethod
    def _pre_filter_signal(
        signal: Dict[str, Any],
        mission_context: Dict[str, Any],
        existing_positions: List[Dict[str, Any]],
        account: Dict[str, Any],
    ) -> Optional[str]:
        """
        Fast, deterministic checks that skip the LLM call for obvious rejections.
        Returns a rejection reason string, or None if the LLM should decide.
        """
        # 1) Portfolio concentration: reject if already >= 2 open positions
        if len(existing_positions) >= 2:
            return "max_positions_reached"

        # 2) Insufficient margin: reject if withdrawable < 5% of account value
        withdrawable = account.get("withdrawable", 0)
        account_value = account.get("account_value", 0)
        if account_value > 0 and withdrawable / account_value < 0.05:
            return "insufficient_margin"

        # 3) Mission lifecycle: early phase (day 1-2) should be conservative — reject LOW confidence
        mission_day = mission_context.get("mission_day", 1)
        if mission_day <= 2 and signal.get("confidence") == "LOW":
            return "low_confidence_early_mission"

        # 4) Late mission wind-down: last 5 days, reject new entries to avoid being stuck
        days_remaining = mission_context.get("days_remaining", 30)
        if days_remaining <= 5:
            return "mission_winding_down"

        # 5) Risk level vs leverage mismatch: CONSERVATIVE missions reject leverage > 1
        risk_level = mission_context.get("risk_level", "MODERATE")
        rec_leverage = signal.get("recommended_leverage", 1)
        if risk_level == "CONSERVATIVE" and rec_leverage > 1:
            return "leverage_exceeds_conservative_limit"

        # 6) Heavy drawdown guard: if mission PnL < -15%, skip LOW confidence signals
        total_pnl_percent = mission_context.get("total_pnl_percent", 0)
        if total_pnl_percent < -15 and signal.get("confidence") != "HIGH":
            return "drawdown_guard"

        # 7) Correlation bucket: reject if adding this asset exceeds bucket cap
        from app.config import get_settings as _get_settings
        _settings = _get_settings()
        if _settings.correlation_bucketing_enabled:
            from app.services.risk_manager import check_correlation_bucket_exceeded
            bucket_caps = {
                "btc_correlated": _settings.correlation_bucket_btc_cap,
                "eth_correlated": _settings.correlation_bucket_eth_cap,
                "uncorrelated": _settings.correlation_bucket_uncorrelated_cap,
            }
            exceeded, reason = check_correlation_bucket_exceeded(
                new_asset=signal["asset"],
                new_leverage=signal.get("recommended_leverage", 1),
                new_margin=account.get("withdrawable", 0) * 0.1,  # estimate 10% allocation
                existing_positions=existing_positions,
                account_value=account.get("account_value", 0),
                max_mission_leverage=mission_context.get("max_leverage", 2),
                bucket_caps=bucket_caps,
            )
            if exceeded:
                return reason

        return None

    async def _execute_trade(
        self,
        hl_client: Any,
        wallet_bridge: Any,
        mission_id: str,
        asset: str,
        is_buy: bool,
        size: float,
        price: float,
        reduce_only: bool,
    ) -> Dict[str, Any]:
        """
        Execute a single trade: build EIP-712 → sign with agent key → submit.
        Uses the per-mission agent key (fast path, zero Turnkey latency).
        """
        try:
            # Build EIP-712 typed data for the order
            typed_data = hl_client.build_eip712_order(
                asset=asset,
                is_buy=is_buy,
                size=size,
                price=price,
                reduce_only=reduce_only,
            )

            # Sign with per-mission agent key (local, fast)
            sign_result = await wallet_bridge.sign_with_agent_key(
                mission_id=mission_id,
                typed_data=typed_data,
            )

            if not sign_result.get("success"):
                return {"success": False, "error": sign_result.get("error", "Signing failed")}

            # Submit to Hyperliquid — place_order_eip712 expects (typed_data, signature)
            submit_result = await hl_client.place_order_eip712(
                typed_data=typed_data,
                signature=sign_result["signature"],
            )

            return submit_result

        except Exception as e:
            logger.error(
                "Trade execution failed",
                mission_id=mission_id,
                asset=asset,
                error=str(e),
            )
            return {"success": False, "error": str(e)}

    async def _record_trade(
        self,
        mission_id: str,
        signal: Dict[str, Any],
        market_price: float,
        position_size: float,
        leverage: int,
        mission_day: int,
        current_value: float,
        result: Dict[str, Any],
        started_at: Optional[str] = None,
    ) -> None:
        """Record trade execution in the database."""
        from app.services.database import record_trade_execution

        try:
            signal_id = f"sig_{datetime.utcnow().strftime('%Y%m%d_%H%M')}_{signal['asset'][:3]}"

            await record_trade_execution(
                mission_id=mission_id,
                action=f"ENTER_{signal['direction']}",
                asset=signal["asset"],
                quantity=position_size,
                price=market_price,
                mission_day=mission_day,
                decision_reason=f"{signal.get('strategy_tag', '')}: {signal.get('reasoning', '')}",
                user_balance=current_value,
                success=result.get("success", False),
                tx_hash=result.get("tx_hash"),
                error_message=result.get("error"),
                signal_id=signal_id,
            )
        except Exception as e:
            logger.error("Failed to record trade", mission_id=mission_id, error=str(e))


# ==================
# Module-level helpers
# ==================

_redis_pool: Optional[Redis] = None


async def get_redis() -> Redis:
    """Get or create the shared Redis connection."""
    global _redis_pool
    if _redis_pool is None:
        settings = get_settings()
        _redis_pool = Redis.from_url(
            settings.redis_url,
            password=settings.redis_password,
            decode_responses=True,
        )
    return _redis_pool


async def close_redis() -> None:
    """Close the Redis connection and arq pool."""
    global _redis_pool, _arq_pool
    if _redis_pool is not None:
        await _redis_pool.close()
        _redis_pool = None
    if _arq_pool is not None:
        await _arq_pool.close()
        _arq_pool = None


# ==================
# arq Connection Pool
# ==================

_arq_pool = None


async def get_arq_pool():
    """Get or create the arq Redis connection pool for enqueuing jobs."""
    global _arq_pool
    if _arq_pool is None:
        from arq.connections import create_pool, RedisSettings
        settings = get_settings()
        # Parse redis URL for arq RedisSettings (supports redis:// and rediss://)
        from urllib.parse import urlparse
        url = settings.redis_url
        parsed = urlparse(url)
        host = parsed.hostname or "redis-cache"
        port = parsed.port or 6379
        password = parsed.password or settings.redis_password
        use_ssl = parsed.scheme == "rediss"
        _arq_pool = await create_pool(
            RedisSettings(host=host, port=port, password=password, ssl=use_ssl)
        )
    return _arq_pool


async def run_trading_cycle(trigger_type: str = "scheduled") -> Dict[str, Any]:
    """
    Run a complete trading cycle: Signal Provider → Dispatcher → Workers.

    This replaces the old monolithic LangGraph workflow for trading.
    Called by the scheduler every 15 minutes.

    Args:
        trigger_type: What triggered this cycle

    Returns:
        Combined results from all phases
    """
    from app.services.signal_provider import generate_market_state

    settings = get_settings()
    redis = await get_redis()

    logger.info("Starting trading cycle", trigger_type=trigger_type)
    cycle_start = time.monotonic()

    # Phase 1: Signal Provider (Brain) - one DeepSeek call
    market_state = await generate_market_state(trigger_type=trigger_type)

    if not market_state.signals:
        logger.info(
            "No signals generated, skipping execution",
            cycle_id=market_state.cycle_id,
        )
        return {
            "cycle_id": market_state.cycle_id,
            "phase": "signal_provider",
            "signals": 0,
            "reason": "no_signals",
        }

    # Phase 2: Dispatcher - fan out to N missions
    dispatcher = ExecutionDispatcher(redis)
    dispatch_result = await dispatcher.dispatch(market_state)

    if dispatch_result["jobs_created"] == 0:
        return {
            "cycle_id": market_state.cycle_id,
            "phase": "dispatcher",
            "signals": len(market_state.signals),
            "jobs_created": 0,
            "reason": dispatch_result.get("reason", "no_missions"),
        }

    # Phase 3: Workers (Hands) - process all jobs
    worker_pool = ExecutionWorkerPool(redis, concurrency=settings.execution_concurrency)
    execution_result = await worker_pool.process_cycle(timeout=90.0)

    total_duration = time.monotonic() - cycle_start

    result = {
        "cycle_id": market_state.cycle_id,
        "signals_generated": len(market_state.signals),
        "jobs_dispatched": dispatch_result["jobs_created"],
        "orders_executed": execution_result["executed"],
        "orders_failed": execution_result["failed"],
        "orders_skipped": execution_result["skipped"],
        "total_duration_seconds": round(total_duration, 2),
        "errors": market_state.errors,
    }

    logger.info("Trading cycle complete", **result)

    # Update cycle stats in Redis
    await redis.hset(
        f"{CYCLE_STATS_KEY}:{market_state.cycle_id}",
        mapping={
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat(),
            "orders_executed": str(execution_result["executed"]),
            "orders_failed": str(execution_result["failed"]),
            "total_duration": str(round(total_duration, 2)),
        },
    )

    return result


async def get_queue_stats() -> Dict[str, Any]:
    """Get current execution queue statistics."""
    redis = await get_redis()

    queue_length = await redis.llen(QUEUE_KEY)
    processing_count = await redis.llen(PROCESSING_KEY)

    # Get latest market state info
    raw_state = await redis.get(MARKET_STATE_KEY)
    latest_cycle = None
    if raw_state:
        state = MarketState.from_json(raw_state)
        latest_cycle = {
            "cycle_id": state.cycle_id,
            "triggered_at": state.triggered_at,
            "signals_count": len(state.signals),
        }

    return {
        "queue_length": queue_length,
        "processing": processing_count,
        "latest_cycle": latest_cycle,
    }

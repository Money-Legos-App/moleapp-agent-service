"""
Circuit Breaker Service
Redis-backed failure tracking that survives restarts and redeploys.
"""

import json
import time
from typing import Dict, List, Optional

import structlog

from app.config import get_settings

logger = structlog.get_logger(__name__)

# Redis key prefixes
CB_FAILURES_KEY = "agent:cb:failures"    # ZSET per mission (timestamps)
CB_TRIPPED_KEY = "agent:cb:tripped"      # HASH mission_id -> trip_timestamp


class ExecutionCircuitBreaker:
    """
    Circuit breaker for execution failures.

    Tracks failures per mission in Redis and "trips" the circuit when
    too many failures occur within a time window.

    When tripped, the mission is skipped until the circuit resets.

    State is persisted in Redis so it survives process restarts.
    """

    def __init__(
        self,
        failure_threshold: Optional[int] = None,
        window_minutes: Optional[int] = None,
        reset_minutes: Optional[int] = None,
    ):
        settings = get_settings()

        self.failure_threshold = failure_threshold or settings.circuit_breaker_failure_threshold
        self.window_seconds = (window_minutes or settings.circuit_breaker_window_minutes) * 60
        self.reset_seconds = (reset_minutes or settings.circuit_breaker_reset_minutes) * 60

        self._redis = None

        logger.info(
            "Circuit breaker initialized",
            failure_threshold=self.failure_threshold,
            window_minutes=self.window_seconds / 60,
            reset_minutes=self.reset_seconds / 60,
        )

    async def _get_redis(self):
        if self._redis is None:
            from app.services.execution_queue import get_redis
            self._redis = await get_redis()
        return self._redis

    def _failures_key(self, mission_id: str) -> str:
        return f"{CB_FAILURES_KEY}:{mission_id}"

    async def record_failure(self, mission_id: str, error: Optional[str] = None) -> bool:
        """
        Record a failure for a mission. Returns True if the circuit has tripped.
        """
        redis = await self._get_redis()
        now = time.time()
        cutoff = now - self.window_seconds
        fkey = self._failures_key(mission_id)

        pipe = redis.pipeline()
        # Remove old failures outside the window
        pipe.zremrangebyscore(fkey, 0, cutoff)
        # Add new failure
        pipe.zadd(fkey, {f"{now}:{error or ''}": now})
        # Count recent failures
        pipe.zcard(fkey)
        # Set TTL so keys auto-expire
        pipe.expire(fkey, self.window_seconds + 60)
        results = await pipe.execute()

        failure_count = results[2]

        logger.warning(
            "Execution failure recorded",
            mission_id=mission_id,
            failure_count=failure_count,
            threshold=self.failure_threshold,
            error=error,
        )

        # Check if circuit should trip
        if failure_count >= self.failure_threshold:
            # Only set tripped if not already tripped
            was_set = await redis.hsetnx(CB_TRIPPED_KEY, mission_id, str(now))
            if was_set:
                logger.error(
                    "CIRCUIT BREAKER TRIPPED",
                    mission_id=mission_id,
                    failure_count=failure_count,
                    reset_in_seconds=self.reset_seconds,
                )
            return True

        return False

    async def record_success(self, mission_id: str) -> None:
        """Record a success — clears failures and resets trip state."""
        redis = await self._get_redis()
        pipe = redis.pipeline()
        pipe.delete(self._failures_key(mission_id))
        pipe.hdel(CB_TRIPPED_KEY, mission_id)
        await pipe.execute()

    async def is_tripped(self, mission_id: str) -> bool:
        """
        Check if the circuit is tripped for a mission.
        Auto-resets if the reset duration has passed.
        """
        redis = await self._get_redis()
        trip_time_str = await redis.hget(CB_TRIPPED_KEY, mission_id)

        if trip_time_str is None:
            return False

        trip_time = float(trip_time_str)
        now = time.time()

        if now - trip_time >= self.reset_seconds:
            # Auto-reset
            pipe = redis.pipeline()
            pipe.hdel(CB_TRIPPED_KEY, mission_id)
            pipe.delete(self._failures_key(mission_id))
            await pipe.execute()

            tripped_duration_min = (now - trip_time) / 60
            logger.warning(
                "CIRCUIT BREAKER AUTO-RESET — trading resumed for mission",
                mission_id=mission_id,
                was_tripped_for_minutes=round(tripped_duration_min, 1),
                tripped_at=trip_time,
                reset_at=now,
            )

            # Record auto-reset in DB for ops visibility
            try:
                from app.services.database import record_agent_audit
                import asyncio
                asyncio.ensure_future(record_agent_audit(
                    node="risk",
                    action="circuit_breaker_auto_reset",
                    mission_id=mission_id,
                    metadata={
                        "tripped_duration_minutes": round(tripped_duration_min, 1),
                        "reset_seconds_config": self.reset_seconds,
                    },
                    success=True,
                ))
            except Exception:
                pass

            return False

        return True

    async def manual_reset(self, mission_id: str) -> bool:
        """Manually reset a tripped circuit."""
        redis = await self._get_redis()
        removed = await redis.hdel(CB_TRIPPED_KEY, mission_id)
        if removed:
            await redis.delete(self._failures_key(mission_id))
            logger.info("Circuit breaker manually reset", mission_id=mission_id)
            return True
        return False

    async def get_status(self, mission_id: str) -> Dict:
        """Get the circuit breaker status for a mission."""
        redis = await self._get_redis()
        now = time.time()
        cutoff = now - self.window_seconds

        pipe = redis.pipeline()
        pipe.zcount(self._failures_key(mission_id), cutoff, "+inf")
        pipe.hget(CB_TRIPPED_KEY, mission_id)
        results = await pipe.execute()

        failure_count = results[0]
        trip_time_str = results[1]

        is_tripped = False
        reset_at = None
        tripped_at = None

        if trip_time_str:
            trip_time = float(trip_time_str)
            if now - trip_time < self.reset_seconds:
                is_tripped = True
                tripped_at = trip_time
                reset_at = trip_time + self.reset_seconds

        return {
            "mission_id": mission_id,
            "failure_count": failure_count,
            "failure_threshold": self.failure_threshold,
            "is_tripped": is_tripped,
            "tripped_at": tripped_at,
            "reset_at": reset_at,
            "window_minutes": self.window_seconds / 60,
        }

    async def get_all_tripped(self) -> List[str]:
        """Get all currently tripped mission IDs."""
        redis = await self._get_redis()
        all_entries = await redis.hgetall(CB_TRIPPED_KEY)
        now = time.time()

        tripped = []
        expired = []
        for mission_id, trip_time_str in all_entries.items():
            if now - float(trip_time_str) < self.reset_seconds:
                tripped.append(mission_id)
            else:
                expired.append(mission_id)

        # Clean up expired entries
        if expired:
            pipe = redis.pipeline()
            for m in expired:
                pipe.hdel(CB_TRIPPED_KEY, m)
                pipe.delete(self._failures_key(m))
            await pipe.execute()

        return tripped

    async def get_stats(self) -> Dict:
        """Get overall circuit breaker statistics."""
        tripped_missions = await self.get_all_tripped()

        return {
            "tripped_count": len(tripped_missions),
            "tripped_missions": tripped_missions,
            "failure_threshold": self.failure_threshold,
            "window_minutes": self.window_seconds / 60,
            "reset_minutes": self.reset_seconds / 60,
        }


# Global circuit breaker instance
_circuit_breaker: Optional[ExecutionCircuitBreaker] = None


def get_circuit_breaker() -> ExecutionCircuitBreaker:
    """Get the global circuit breaker instance."""
    global _circuit_breaker
    if _circuit_breaker is None:
        _circuit_breaker = ExecutionCircuitBreaker()
    return _circuit_breaker


def reset_circuit_breaker() -> None:
    """Reset the global circuit breaker (useful for testing)."""
    global _circuit_breaker
    _circuit_breaker = None

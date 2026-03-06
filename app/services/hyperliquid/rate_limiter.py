"""
Hyperliquid Weight-Aware Rate Limiter

Matches Hyperliquid's actual rate limit budget:
- 1200 weight per minute per IP
- Info endpoints: weight 2 (allMids, clearinghouseState, l2Book, etc.) or weight 20 (meta, etc.)
- Exchange endpoints: weight 1 + floor(batch_length / 40)

Uses aiolimiter's AsyncLimiter (leaky bucket) with separate buckets for
info (read) and exchange (write) endpoints sharing a global budget.
This ensures exchange (trading) requests always have reserved capacity.
"""

from enum import Enum
from typing import Optional

import structlog
from aiolimiter import AsyncLimiter
from prometheus_client import Counter, Histogram

logger = structlog.get_logger(__name__)

# Prometheus metrics
HL_REQUEST_WEIGHT = Counter(
    "hl_request_weight_total",
    "Total weight consumed by Hyperliquid API requests",
    ["endpoint_type", "request_type"],
)
HL_RATE_LIMIT_WAIT = Histogram(
    "hl_rate_limit_wait_seconds",
    "Time spent waiting for rate limit token",
    ["endpoint_type"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# Hyperliquid endpoint weight map
# See: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/rate-limits
INFO_WEIGHTS = {
    "allMids": 2,
    "clearinghouseState": 2,
    "l2Book": 2,
    "orderStatus": 2,
    "spotClearinghouseState": 2,
    "exchangeStatus": 2,
    "openOrders": 2,
    "userFills": 2,
    "recentTrades": 2,
    "extraAgents": 2,
}
DEFAULT_INFO_WEIGHT = 20
EXCHANGE_BASE_WEIGHT = 1


class EndpointType(Enum):
    INFO = "info"
    EXCHANGE = "exchange"


class HyperliquidRateLimiter:
    """
    Weight-aware rate limiter for Hyperliquid API.

    Maintains a global weight budget with separate tracking for
    info vs exchange to allow prioritization. Exchange (write)
    operations get a reserved bucket that info reads cannot exhaust.
    """

    def __init__(
        self,
        global_weight_per_minute: float = 1100.0,
        info_weight_per_minute: float = 900.0,
        exchange_weight_per_minute: float = 200.0,
    ):
        self._global = AsyncLimiter(global_weight_per_minute, 60.0)
        self._info = AsyncLimiter(info_weight_per_minute, 60.0)
        self._exchange = AsyncLimiter(exchange_weight_per_minute, 60.0)

    @staticmethod
    def get_weight(
        endpoint_type: EndpointType,
        request_type: str,
        batch_length: int = 1,
    ) -> int:
        """Calculate the weight for a request."""
        if endpoint_type == EndpointType.EXCHANGE:
            return EXCHANGE_BASE_WEIGHT + (batch_length // 40)
        return INFO_WEIGHTS.get(request_type, DEFAULT_INFO_WEIGHT)

    async def acquire(
        self,
        endpoint_type: EndpointType,
        request_type: str,
        batch_length: int = 1,
    ) -> None:
        """
        Acquire rate limit tokens for a request.
        Blocks until capacity is available. Acquires from both the
        endpoint-specific bucket AND the global bucket.
        """
        import time

        weight = self.get_weight(endpoint_type, request_type, batch_length)
        start = time.monotonic()

        if endpoint_type == EndpointType.INFO:
            await self._info.acquire(weight)
        else:
            await self._exchange.acquire(weight)

        await self._global.acquire(weight)

        elapsed = time.monotonic() - start
        HL_REQUEST_WEIGHT.labels(
            endpoint_type=endpoint_type.value,
            request_type=request_type,
        ).inc(weight)
        HL_RATE_LIMIT_WAIT.labels(
            endpoint_type=endpoint_type.value,
        ).observe(elapsed)

        if elapsed > 1.0:
            logger.info(
                "Rate limiter delayed request",
                endpoint_type=endpoint_type.value,
                request_type=request_type,
                weight=weight,
                wait_seconds=round(elapsed, 2),
            )


# Module-level singleton (shared across all HyperliquidClient instances)
_shared_rate_limiter: Optional[HyperliquidRateLimiter] = None


def get_hl_rate_limiter() -> HyperliquidRateLimiter:
    """Get the shared rate limiter singleton."""
    global _shared_rate_limiter
    if _shared_rate_limiter is None:
        from app.config import get_settings

        settings = get_settings()
        global_budget = settings.hl_weight_budget_per_minute
        info_budget = settings.hl_info_weight_budget
        exchange_budget = settings.hl_exchange_weight_budget

        _shared_rate_limiter = HyperliquidRateLimiter(
            global_weight_per_minute=global_budget,
            info_weight_per_minute=info_budget,
            exchange_weight_per_minute=exchange_budget,
        )
        logger.info(
            "Rate limiter initialized",
            global_budget=global_budget,
            info_budget=info_budget,
            exchange_budget=exchange_budget,
        )
    return _shared_rate_limiter

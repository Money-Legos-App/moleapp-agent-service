"""
arq Worker Settings

Entry point for the arq worker process:
    python -m arq app.workers.arq_settings.WorkerSettings

Runs as a separate container (agent-worker) alongside the FastAPI server.
Shares the same Redis, Postgres, and Hyperliquid API connections.
"""

from arq.connections import RedisSettings

from app.workers.pnl_worker import update_pnl_for_mission
from app.workers.deposit_worker import check_deposit_for_mission


async def on_startup(ctx: dict) -> None:
    """Initialize shared resources for all arq workers."""
    import structlog
    from app.config import get_settings
    from app.services.execution_queue import get_redis, RateLimiter

    logger = structlog.get_logger("arq.worker")
    settings = get_settings()

    # Shared Redis connection
    redis = await get_redis()
    ctx["redis"] = redis

    # Shared rate limiter (uses hardcoded RATE_LIMIT_KEY internally)
    ctx["rate_limiter"] = RateLimiter(
        redis=redis,
        max_per_second=settings.hl_rate_limit_per_second,
    )

    logger.info(
        "arq worker started",
        max_jobs=settings.arq_max_jobs,
        rate_limit=settings.hl_rate_limit_per_second,
    )


async def on_shutdown(ctx: dict) -> None:
    """Cleanup shared resources."""
    import structlog
    from app.services.execution_queue import close_redis

    logger = structlog.get_logger("arq.worker")
    await close_redis()
    logger.info("arq worker shut down")


def _get_redis_settings() -> RedisSettings:
    """Build arq RedisSettings from app config."""
    from app.config import get_settings
    settings = get_settings()
    from urllib.parse import urlparse
    url = settings.redis_url
    parsed = urlparse(url)
    host = parsed.hostname or "redis-cache"
    port = parsed.port or 6379
    password = parsed.password or settings.redis_password
    use_ssl = parsed.scheme == "rediss"
    return RedisSettings(host=host, port=port, password=password, ssl=use_ssl)


class WorkerSettings:
    """arq worker configuration."""

    redis_settings = _get_redis_settings()

    functions = [
        update_pnl_for_mission,
        check_deposit_for_mission,
    ]

    on_startup = on_startup
    on_shutdown = on_shutdown

    # Read concurrency from env (ARQ_MAX_JOBS, default 3 for 512MB plans)
    from app.config import get_settings as _gs
    max_jobs = _gs().arq_max_jobs
    job_timeout = _gs().arq_job_timeout_seconds
    max_tries = 30  # HL bridge can take 5-10+ minutes; allow plenty of retries
    retry_jobs = True
    allow_abort_jobs = True

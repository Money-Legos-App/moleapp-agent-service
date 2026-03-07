"""
Alkebulan Agent Service
AI-powered DeFi trading agent for MoleApp - Hyperliquid perpetuals via RAG + DeepSeek
"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from app.config import get_settings, HyperliquidConfig

# Ensure stdlib logging has a handler before structlog wraps it
logging.basicConfig(format="%(message)s", stream=sys.stdout, level=logging.INFO)

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)
settings = get_settings()

# Prometheus metrics
REQUEST_COUNT = Counter(
    "agent_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "agent_http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
)
SIGNAL_COUNT = Counter(
    "agent_signals_generated_total",
    "Total trading signals generated",
    ["asset", "direction"],
)
TRADE_COUNT = Counter(
    "agent_trades_executed_total",
    "Total trades executed",
    ["asset", "direction", "success"],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown events."""
    # Startup
    logger.info(
        "Starting Alkebulan Agent Service",
        version=settings.service_version,
        environment=settings.environment,
        hyperliquid_testnet=settings.is_testnet,
    )

    # Initialize services
    try:
        from app.tasks.scheduler import AgentScheduler

        # Initialize FAISS store (skip if DISABLE_RAG=true to save memory on 512MB plans)
        if settings.disable_rag:
            app.state.faiss_store = None
            logger.info("RAG/FAISS disabled via DISABLE_RAG=true — saving ~300MB RAM")
        else:
            from app.services.rag.faiss_store import FAISSStore
            faiss_store = FAISSStore()
            await faiss_store.initialize()
            app.state.faiss_store = faiss_store
            logger.info("FAISS store initialized", index_path=settings.faiss_index_path)

        # Initialize scheduler (disabled in dev by default)
        scheduler = AgentScheduler()
        if settings.environment != "development" or settings.debug:
            await scheduler.start()
            app.state.scheduler = scheduler
            logger.info(
                "Scheduler started",
                analysis_interval=f"{settings.analysis_interval_minutes}m",
                pnl_sync_interval=f"{settings.pnl_sync_interval_seconds}s",
            )
        else:
            app.state.scheduler = None
            logger.info("Scheduler disabled in development mode")

        # ── OPERATING MODE BANNER ──────────────────────────────────
        network = "MAINNET" if settings.hyperliquid_mainnet else "TESTNET"
        mode = "PAPER TRADING" if settings.dry_run else "LIVE TRADING"
        vault_ok = settings.vault_configured

        if not settings.dry_run and settings.hyperliquid_mainnet:
            logger.critical(
                f"*** {mode} on {network} — REAL FUNDS AT RISK ***",
                dry_run=False,
                hyperliquid_api=HyperliquidConfig.get_api_url(True),
                vault_configured=vault_ok,
                scheduler_active=app.state.scheduler is not None,
            )
        elif settings.dry_run:
            logger.info(
                f"{mode} on {network}",
                dry_run=True,
                hyperliquid_api=HyperliquidConfig.get_api_url(settings.hyperliquid_mainnet),
                scheduler_active=app.state.scheduler is not None,
                allowed_assets=settings.allowed_assets,
            )
        else:
            logger.warning(
                f"{mode} on {network}",
                dry_run=False,
                hyperliquid_api=HyperliquidConfig.get_api_url(settings.hyperliquid_mainnet),
                scheduler_active=app.state.scheduler is not None,
            )

        # Start WebSocket market data feed (real-time price updates)
        app.state.ws_feed = None
        if settings.ws_enabled:
            from app.services.hyperliquid.ws_feed import HyperliquidWSFeed
            from app.services.hyperliquid.ws_cache_bridge import update_redis_from_ws_mids
            from app.services.execution_queue import get_redis

            ws_redis = await get_redis()

            async def on_ws_prices(mids: dict):
                await update_redis_from_ws_mids(mids, ws_redis)

            ws_feed = HyperliquidWSFeed(
                is_mainnet=settings.hyperliquid_mainnet,
                on_prices_update=on_ws_prices,
                max_reconnect_delay=settings.ws_reconnect_max_delay,
            )
            await ws_feed.start()
            app.state.ws_feed = ws_feed
            logger.info("WebSocket market data feed started")

            # Start Fast Actor if enabled (requires WS feed)
            app.state.fast_actor = None
            if settings.fast_actor_enabled:
                from app.services.fast_actor import FastActor

                fast_actor = FastActor(ws_redis)
                await fast_actor.start()
                app.state.fast_actor = fast_actor

                # Extend WS callback to also feed the Fast Actor
                original_callback = on_ws_prices

                async def on_ws_prices_with_fast_actor(mids: dict):
                    await original_callback(mids)
                    await fast_actor.on_price_tick(mids)

                ws_feed._on_prices_update = on_ws_prices_with_fast_actor
                logger.info("Fast Actor started and wired to WebSocket feed")
        else:
            logger.info("WebSocket feed disabled (WS_ENABLED=false)")

        # Initialize Langfuse observability
        from app.services.observability.langfuse_client import get_langfuse
        lf_client = get_langfuse()
        logger.info(
            "Langfuse status",
            enabled=lf_client.enabled,
            host=settings.langfuse_host if lf_client.enabled else "n/a",
        )

        # Start embedded arq worker (background job processor)
        # Replaces the separate agent-worker container
        from arq.worker import create_worker
        from app.workers.arq_settings import WorkerSettings

        worker = create_worker(WorkerSettings)
        worker_task = asyncio.create_task(worker.async_run())
        app.state.arq_worker = worker
        app.state.arq_worker_task = worker_task
        logger.info(
            "Embedded arq worker started",
            max_jobs=WorkerSettings.max_jobs,
            functions=[f.__name__ for f in WorkerSettings.functions],
        )

    except Exception as e:
        logger.error("Failed to initialize services", error=str(e))
        # Don't fail startup - allow health checks to report unhealthy
        app.state.faiss_store = None
        app.state.scheduler = None

    yield

    # Shutdown
    logger.info("Shutting down Alkebulan Agent Service")

    if hasattr(app.state, "scheduler") and app.state.scheduler:
        await app.state.scheduler.stop()
        logger.info("Scheduler stopped")

    # Stop Fast Actor (before WS feed)
    if hasattr(app.state, "fast_actor") and app.state.fast_actor:
        await app.state.fast_actor.stop()
        logger.info("Fast Actor stopped")

    # Stop WebSocket feed
    if hasattr(app.state, "ws_feed") and app.state.ws_feed:
        await app.state.ws_feed.stop()
        logger.info("WebSocket feed stopped")

    # Stop embedded arq worker
    if hasattr(app.state, "arq_worker") and app.state.arq_worker:
        await app.state.arq_worker.close()
        if hasattr(app.state, "arq_worker_task"):
            app.state.arq_worker_task.cancel()
        logger.info("Embedded arq worker stopped")

    # Flush Langfuse pending events
    try:
        from app.services.observability.langfuse_client import get_langfuse
        get_langfuse().shutdown()
        logger.info("Langfuse flushed and shut down")
    except Exception:
        pass

    # Clean up arq connection pool
    try:
        from app.services.execution_queue import close_redis
        await close_redis()
    except Exception:
        pass


# Create FastAPI app
app = FastAPI(
    title="Alkebulan Agent Service",
    description="AI-powered DeFi trading agent for MoleApp - Hyperliquid perpetuals via RAG + DeepSeek",
    version=settings.service_version,
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if not settings.is_production else ["https://moleapp.io"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests and track metrics."""
    import time

    start_time = time.time()

    response = await call_next(request)

    duration = time.time() - start_time
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code,
    ).inc()
    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path,
    ).observe(duration)

    logger.info(
        "Request completed",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=round(duration * 1000, 2),
    )

    return response


# Response wrapper middleware — mobile expects { status, data } envelope
@app.middleware("http")
async def wrap_api_response(request: Request, call_next):
    """Wrap /api/ JSON responses in { status, data } for mobile compatibility."""
    import json as _json

    response = await call_next(request)

    # Only wrap API routes (skip health, metrics, docs, etc.)
    if not request.url.path.startswith("/api/"):
        return response

    # Only wrap JSON responses
    content_type = response.headers.get("content-type", "")
    if "application/json" not in content_type:
        return response

    # Read the streaming response body
    body_bytes = b""
    async for chunk in response.body_iterator:
        body_bytes += chunk

    try:
        data = _json.loads(body_bytes)
    except (ValueError, UnicodeDecodeError):
        return Response(content=body_bytes, status_code=response.status_code,
                        media_type="application/json")

    # Wrap based on status code
    if 200 <= response.status_code < 400:
        wrapped = {"status": "ok", "data": data}
    else:
        msg = ""
        if isinstance(data, dict):
            msg = data.get("detail") or data.get("message") or data.get("error") or "Unknown error"
        wrapped = {"status": "error", "message": msg, "data": None}

    return JSONResponse(content=wrapped, status_code=response.status_code)


# Health check endpoint
@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint for Docker/Kubernetes."""
    faiss_healthy = (
        hasattr(app.state, "faiss_store")
        and app.state.faiss_store is not None
        and app.state.faiss_store.is_ready
    )

    # FAISS not required when RAG is disabled
    rag_ok = faiss_healthy or settings.disable_rag

    scheduler_healthy = (
        not settings.is_production
        or (
            hasattr(app.state, "scheduler")
            and app.state.scheduler is not None
            and app.state.scheduler.is_running
        )
    )

    ws_healthy = (
        hasattr(app.state, "ws_feed")
        and app.state.ws_feed is not None
        and app.state.ws_feed.is_healthy
    )

    is_healthy = rag_ok or settings.environment == "development"

    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "service": settings.service_name,
        "version": settings.service_version,
        "environment": settings.environment,
        "mode": "dry_run" if settings.dry_run else "live",
        "network": "mainnet" if settings.hyperliquid_mainnet else "testnet",
        "checks": {
            "faiss": "disabled" if settings.disable_rag else ("ok" if faiss_healthy else "not_ready"),
            "scheduler": "ok" if scheduler_healthy else "not_running",
            "ws_feed": "ok" if ws_healthy else "disconnected",
        },
    }


# Readiness check (for k8s)
@app.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """Readiness check - service is ready to accept traffic."""
    faiss_ready = (
        hasattr(app.state, "faiss_store")
        and app.state.faiss_store is not None
        and app.state.faiss_store.is_ready
    )

    return {
        "ready": faiss_ready or settings.disable_rag or settings.environment == "development",
        "faiss_index_loaded": faiss_ready,
        "rag_disabled": settings.disable_rag,
    }


# Prometheus metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Root endpoint
@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint with service info."""
    return {
        "service": "Alkebulan Agent Service",
        "version": settings.service_version,
        "description": "AI-powered DeFi trading agent for MoleApp",
        "docs": "/docs" if not settings.is_production else "disabled",
    }


# Include API routers
from app.api.v1 import router as api_v1_router
app.include_router(api_v1_router, prefix="/api/v1/agent")


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(
        "Unhandled exception",
        path=request.url.path,
        method=request.method,
        error=str(exc),
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if settings.debug else "An unexpected error occurred",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.environment == "development",
        log_level=settings.log_level.lower(),
    )

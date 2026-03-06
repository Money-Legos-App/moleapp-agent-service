"""
Admin API Endpoints
Manual triggers and system management
"""

from datetime import datetime
from typing import Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.api.deps import get_internal_auth

logger = structlog.get_logger(__name__)
router = APIRouter()


# ==================
# Request/Response Models
# ==================

class TriggerWorkflowRequest(BaseModel):
    """Request to manually trigger a workflow."""

    trigger_type: str = "manual"


class WorkflowStatusResponse(BaseModel):
    """Workflow execution status."""

    status: str
    triggered_at: datetime
    completed_at: Optional[datetime]
    signals_generated: int
    orders_executed: int
    orders_failed: int
    errors: list


class SchedulerStatusResponse(BaseModel):
    """Scheduler status."""

    running: bool
    jobs: list


class IndexStatsResponse(BaseModel):
    """FAISS index statistics."""

    is_ready: bool
    num_vectors: int
    num_patterns: int
    index_path: str
    embedding_model: str
    assets: list


# ==================
# Endpoints (Internal only)
# ==================

@router.post("/trigger-analysis", response_model=WorkflowStatusResponse)
async def trigger_analysis(
    request: TriggerWorkflowRequest,
    _: bool = Depends(get_internal_auth),
):
    """
    Manually trigger the trading cycle.

    Runs: Signal Provider (Brain) → Dispatcher → Workers (Hands)
    """
    from app.services.execution_queue import run_trading_cycle

    logger.info("Manually triggering trading cycle")

    try:
        result = await run_trading_cycle(trigger_type=request.trigger_type)

        return WorkflowStatusResponse(
            status="completed",
            triggered_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            signals_generated=result.get("signals_generated", 0),
            orders_executed=result.get("orders_executed", 0),
            orders_failed=result.get("orders_failed", 0),
            errors=result.get("errors", []),
        )

    except Exception as e:
        logger.error("Trading cycle trigger failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Trading cycle failed: {str(e)}",
        )


@router.post("/trigger-pnl-sync")
async def trigger_pnl_sync(
    _: bool = Depends(get_internal_auth),
):
    """
    Manually trigger PnL synchronization from Hyperliquid.
    """
    from app.tasks.pnl_sync import sync_positions_from_hyperliquid

    logger.info("Manually triggering PnL sync")

    try:
        updated_count = await sync_positions_from_hyperliquid()
        return {
            "status": "completed",
            "positions_updated": updated_count,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error("PnL sync trigger failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PnL sync failed: {str(e)}",
        )


@router.post("/trigger-daily-snapshot")
async def trigger_daily_snapshot(
    _: bool = Depends(get_internal_auth),
):
    """
    Manually trigger daily PnL snapshot creation.
    """
    from app.tasks.pnl_sync import create_daily_snapshots

    logger.info("Manually triggering daily snapshot")

    try:
        snapshot_count = await create_daily_snapshots()
        return {
            "status": "completed",
            "snapshots_created": snapshot_count,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error("Daily snapshot trigger failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Daily snapshot failed: {str(e)}",
        )


@router.get("/scheduler/status", response_model=SchedulerStatusResponse)
async def get_scheduler_status(
    _: bool = Depends(get_internal_auth),
):
    """
    Get current scheduler status and job information.
    """
    from fastapi import Request
    from app.main import app

    scheduler = getattr(app.state, "scheduler", None)

    if scheduler is None:
        return SchedulerStatusResponse(
            running=False,
            jobs=[],
        )

    status = scheduler.get_job_status()
    return SchedulerStatusResponse(
        running=status.get("running", False),
        jobs=status.get("jobs", []),
    )


@router.post("/scheduler/trigger-job/{job_id}")
async def trigger_scheduler_job(
    job_id: str,
    _: bool = Depends(get_internal_auth),
):
    """
    Manually trigger a specific scheduler job.

    Valid job IDs: market_analysis, pnl_sync, daily_snapshot, position_monitoring, mission_expiry
    """
    from app.main import app

    scheduler = getattr(app.state, "scheduler", None)

    if scheduler is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scheduler not running",
        )

    success = await scheduler.trigger_job(job_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found",
        )

    return {
        "status": "triggered",
        "job_id": job_id,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/faiss/stats", response_model=IndexStatsResponse)
async def get_faiss_stats(
    _: bool = Depends(get_internal_auth),
):
    """
    Get FAISS vector store statistics.
    """
    from app.main import app

    faiss_store = getattr(app.state, "faiss_store", None)

    if faiss_store is None:
        return IndexStatsResponse(
            is_ready=False,
            num_vectors=0,
            num_patterns=0,
            index_path="",
            embedding_model="",
            assets=[],
        )

    stats = faiss_store.get_stats()
    return IndexStatsResponse(**stats)


@router.post("/faiss/rebuild")
async def rebuild_faiss_index(
    days: int = 365,
    _: bool = Depends(get_internal_auth),
):
    """
    Rebuild the FAISS index from historical data.

    This fetches fresh market data and rebuilds the pattern index.
    """
    from app.services.rag import FAISSStore, MarketDataLoader
    from app.config import get_settings

    logger.info("Rebuilding FAISS index", days=days)

    settings = get_settings()

    try:
        # Build new index
        data_loader = MarketDataLoader()
        all_patterns = await data_loader.build_full_index(
            assets=settings.allowed_assets,
            days=days,
        )

        # Initialize new store and add patterns
        faiss_store = FAISSStore()
        await faiss_store.initialize()
        added = await faiss_store.add_patterns(all_patterns)

        return {
            "status": "completed",
            "patterns_added": added,
            "assets_processed": settings.allowed_assets,
            "days_of_history": days,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error("FAISS rebuild failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Index rebuild failed: {str(e)}",
        )


@router.get("/execution-queue/stats")
async def get_execution_queue_stats(
    _: bool = Depends(get_internal_auth),
):
    """
    Get execution queue statistics (Redis-backed).
    Shows queue depth, latest cycle info, and rate limit state.
    """
    from app.services.execution_queue import get_queue_stats

    try:
        stats = await get_queue_stats()
        return {
            "status": "ok",
            **stats,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error("Failed to get queue stats", error=str(e))
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }


@router.get("/health/detailed")
async def detailed_health(
    _: bool = Depends(get_internal_auth),
):
    """
    Detailed health check with all service statuses.
    """
    from app.main import app
    from app.config import get_settings
    from app.services.hyperliquid import HyperliquidClient

    settings = get_settings()
    health = {
        "service": settings.service_name,
        "version": settings.service_version,
        "environment": settings.environment,
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {},
    }

    # FAISS check
    faiss_store = getattr(app.state, "faiss_store", None)
    health["checks"]["faiss"] = {
        "status": "ok" if faiss_store and faiss_store.is_ready else "not_ready",
        "vectors": faiss_store.get_stats().get("num_vectors", 0) if faiss_store else 0,
    }

    # Scheduler check
    scheduler = getattr(app.state, "scheduler", None)
    health["checks"]["scheduler"] = {
        "status": "ok" if scheduler and scheduler.is_running else "not_running",
    }

    # Redis check
    try:
        from app.services.execution_queue import get_redis
        redis = await get_redis()
        redis_info = await redis.ping()
        queue_len = await redis.llen("agent:execution:queue")
        health["checks"]["redis"] = {
            "status": "ok" if redis_info else "error",
            "queue_length": queue_len,
        }
    except Exception as e:
        health["checks"]["redis"] = {
            "status": "error",
            "error": str(e),
        }

    # Hyperliquid check
    try:
        hl_client = HyperliquidClient()
        prices = await hl_client.get_all_mids()
        await hl_client.close()
        health["checks"]["hyperliquid"] = {
            "status": "ok",
            "assets_available": len(prices),
        }
    except Exception as e:
        health["checks"]["hyperliquid"] = {
            "status": "error",
            "error": str(e),
        }

    return health

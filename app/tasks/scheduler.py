"""
Agent Scheduler
Handles periodic tasks like market analysis, PnL sync, and position monitoring
"""

import asyncio
from datetime import datetime
from typing import Optional

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

logger = structlog.get_logger(__name__)


class AgentScheduler:
    """
    Manages scheduled tasks for the Alkebulan Agent.

    Tasks:
    - Market analysis (Layer A): Every 15 minutes
    - PnL sync: Every 60 seconds
    - Daily PnL snapshot: Once per day at midnight UTC
    - Position monitoring: Every 5 minutes
    """

    def __init__(self):
        """Initialize the scheduler."""
        from app.config import get_settings

        self.settings = get_settings()
        self._scheduler: Optional[AsyncIOScheduler] = None
        self._is_running = False

    @property
    def is_running(self) -> bool:
        """Check if the scheduler is running."""
        return self._is_running

    async def start(self) -> None:
        """Start the scheduler with all configured jobs."""
        if self._is_running:
            logger.warning("Scheduler already running")
            return

        self._scheduler = AsyncIOScheduler()

        # Market Analysis Job (Layer A)
        self._scheduler.add_job(
            self._run_market_analysis,
            trigger=IntervalTrigger(minutes=self.settings.analysis_interval_minutes),
            id="market_analysis",
            name="Market Analysis (Layer A)",
            max_instances=1,
            replace_existing=True,
        )

        # PnL Sync: fetch market data once, cache in Redis, fan out arq jobs per mission
        self._scheduler.add_job(
            self._run_market_data_and_fan_out,
            trigger=IntervalTrigger(seconds=self.settings.pnl_sync_interval_seconds),
            id="pnl_sync",
            name="PnL Sync (Queue Fan-Out)",
            max_instances=1,
            replace_existing=True,
        )

        # Daily PnL Snapshot
        self._scheduler.add_job(
            self._run_daily_snapshot,
            trigger=CronTrigger(hour=self.settings.pnl_snapshot_hour, minute=0),
            id="daily_snapshot",
            name="Daily PnL Snapshot",
            max_instances=1,
            replace_existing=True,
        )

        # Position Monitoring (stop loss, take profit, liquidation alerts)
        self._scheduler.add_job(
            self._run_position_monitoring,
            trigger=IntervalTrigger(minutes=5),
            id="position_monitoring",
            name="Position Monitoring",
            max_instances=1,
            replace_existing=True,
        )

        # Mission Expiry Check (check for missions ending soon)
        self._scheduler.add_job(
            self._run_mission_expiry_check,
            trigger=IntervalTrigger(hours=1),
            id="mission_expiry",
            name="Mission Expiry Check",
            max_instances=1,
            replace_existing=True,
        )

        # Backup Risk Monitor (dead-man's-switch heartbeat check)
        self._scheduler.add_job(
            self._run_backup_risk_monitor,
            trigger=IntervalTrigger(seconds=60),
            id="backup_risk_monitor",
            name="Backup Risk Monitor (Heartbeat Check)",
            max_instances=1,
            replace_existing=True,
        )

        # Deposit checks are event-driven (enqueued on mission activation via arq)
        # — no cron polling needed

        self._scheduler.start()
        self._is_running = True

        logger.info(
            "Scheduler started",
            jobs=[job.id for job in self._scheduler.get_jobs()],
            analysis_interval=f"{self.settings.analysis_interval_minutes}m",
            pnl_sync_interval=f"{self.settings.pnl_sync_interval_seconds}s",
        )

    async def stop(self) -> None:
        """Stop the scheduler and clean up Redis connections."""
        if self._scheduler and self._is_running:
            self._scheduler.shutdown(wait=False)
            self._is_running = False

            # Clean up Redis connection pool
            try:
                from app.services.execution_queue import close_redis
                await close_redis()
            except Exception:
                pass

            logger.info("Scheduler stopped")

    async def trigger_job(self, job_id: str) -> bool:
        """Manually trigger a job by ID."""
        if not self._scheduler:
            logger.error("Scheduler not initialized")
            return False

        job = self._scheduler.get_job(job_id)
        if not job:
            logger.error("Job not found", job_id=job_id)
            return False

        # Trigger immediately
        self._scheduler.modify_job(job_id, next_run_time=datetime.now())
        logger.info("Job triggered manually", job_id=job_id)
        return True

    async def _run_market_analysis(self) -> None:
        """
        Run the trading cycle: Signal Provider → Dispatcher → Workers.

        Architecture: "One Brain, Many Hands"
        1. Signal Provider (Brain): One DeepSeek call → MarketState
        2. Dispatcher: Fan out N jobs to Redis queue (one per mission)
        3. Workers (Hands): Process jobs with per-mission agent key signing + rate limiting
        """
        logger.info("Starting trading cycle (Signal → Dispatch → Execute)")

        try:
            from app.services.execution_queue import run_trading_cycle

            result = await run_trading_cycle(trigger_type="scheduled")

            logger.info(
                "Trading cycle completed",
                cycle_id=result.get("cycle_id"),
                signals_generated=result.get("signals_generated", 0),
                jobs_dispatched=result.get("jobs_dispatched", 0),
                orders_executed=result.get("orders_executed", 0),
                orders_failed=result.get("orders_failed", 0),
                duration_seconds=result.get("total_duration_seconds"),
            )

        except Exception as e:
            logger.error(
                "Trading cycle failed",
                error=str(e),
                exc_info=True,
            )

    async def _run_market_data_and_fan_out(self) -> None:
        """
        Queue-based PnL sync: fetch market data once, cache in Redis, fan out jobs.

        1. Calls metaAndAssetCtxs once (1 API call for ALL assets)
        2. Caches prices in Redis hash
        3. Enqueues one arq job per active mission
        4. arq workers compute PnL using cached prices + 1 clearinghouseState call per mission
        """
        try:
            from app.workers.market_data_worker import refresh_market_prices, fan_out_pnl_jobs

            prices = await refresh_market_prices()
            jobs_count = await fan_out_pnl_jobs()

            logger.info(
                "Market data refreshed and PnL jobs dispatched",
                assets_cached=len(prices),
                jobs_dispatched=jobs_count,
            )

        except Exception as e:
            logger.error("Market data / PnL fan-out failed", error=str(e))

    async def _run_daily_snapshot(self) -> None:
        """
        Take a daily PnL snapshot for all active missions.

        Creates AgentPnLSnapshot records for charting and historical tracking.
        """
        logger.info("Taking daily PnL snapshot")

        try:
            from app.tasks.pnl_sync import create_daily_snapshots

            snapshot_count = await create_daily_snapshots()
            logger.info("Daily snapshot completed", snapshots_created=snapshot_count)

        except Exception as e:
            logger.error("Daily snapshot failed", error=str(e))

    async def _run_position_monitoring(self) -> None:
        """
        Monitor positions for risk events.

        Checks:
        - Approaching liquidation
        - Stop loss triggers
        - Take profit triggers
        - Position health alerts
        """
        logger.debug("Running position monitoring")

        try:
            from app.tasks.monitoring import check_positions_for_alerts

            alerts = await check_positions_for_alerts()

            if alerts:
                logger.info(
                    "Position alerts generated",
                    alert_count=len(alerts),
                )

        except Exception as e:
            logger.error("Position monitoring failed", error=str(e))

    async def _run_mission_expiry_check(self) -> None:
        """
        Check for missions approaching expiry.

        Actions:
        - Notify users of upcoming mission end
        - Force close positions on last day
        - Mark completed missions
        """
        logger.debug("Running mission expiry check")

        try:
            from app.tasks.mission_lifecycle import check_mission_expiry

            result = await check_mission_expiry()

            if result.get("ending_soon") or result.get("force_closed"):
                logger.info(
                    "Mission expiry check completed",
                    ending_soon=result.get("ending_soon", 0),
                    force_closed=result.get("force_closed", 0),
                )

        except Exception as e:
            logger.error("Mission expiry check failed", error=str(e))

    async def _run_backup_risk_monitor(self) -> None:
        """
        Dead-man's-switch: check heartbeat and run backup risk evaluation
        if primary monitor is unresponsive.
        """
        try:
            from app.tasks.monitoring import backup_risk_monitor

            result = await backup_risk_monitor()

            if result.get("action") != "none":
                logger.warning(
                    "Backup risk monitor activated",
                    status=result.get("status"),
                    action=result.get("action"),
                )

        except Exception as e:
            logger.error("Backup risk monitor failed", error=str(e))

    def get_job_status(self) -> dict:
        """Get status of all scheduled jobs."""
        if not self._scheduler:
            return {"running": False, "jobs": []}

        jobs = []
        for job in self._scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                "pending": job.pending,
            })

        return {
            "running": self._is_running,
            "jobs": jobs,
        }

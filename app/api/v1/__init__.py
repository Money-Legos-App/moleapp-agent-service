"""API v1 module."""

from fastapi import APIRouter

from . import missions, signals, positions, admin, dashboard, competition

router = APIRouter()

router.include_router(missions.router, prefix="/missions", tags=["missions"])
router.include_router(signals.router, prefix="/signals", tags=["signals"])
router.include_router(positions.router, prefix="/positions", tags=["positions"])
router.include_router(dashboard.router, prefix="/dashboard", tags=["dashboard"])
router.include_router(admin.router, prefix="/admin", tags=["admin"])
router.include_router(competition.router, prefix="/competition", tags=["competition"])

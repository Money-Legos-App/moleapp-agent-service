"""
Signals API Endpoints
View generated trading signals
"""

from datetime import datetime
from typing import List, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from app.api.deps import UserInfo, get_current_user

logger = structlog.get_logger(__name__)
router = APIRouter()


# ==================
# Response Models
# ==================

class SignalResponse(BaseModel):
    """Trading signal response."""

    id: str
    signal_id: str
    asset: str
    direction: str  # LONG or SHORT
    confidence: str  # LOW, MEDIUM, HIGH
    recommended_leverage: int
    strategy_tag: str
    reasoning: str
    max_drawdown_30d: Optional[float] = None
    volatility_score: Optional[float] = None
    generated_at: datetime
    expires_at: datetime
    is_processed: bool
    users_notified: int
    orders_executed: int


class SignalStatsResponse(BaseModel):
    """Signal statistics."""

    total_signals_today: int
    long_signals: int
    short_signals: int
    high_confidence: int
    success_rate_7d: float
    avg_return_per_signal: float


# ==================
# Endpoints
# ==================

@router.get("", response_model=List[SignalResponse])
async def list_signals(
    user: UserInfo = Depends(get_current_user),
    asset: Optional[str] = Query(None, description="Filter by asset"),
    direction: Optional[str] = Query(None, description="Filter by direction: LONG, SHORT"),
    confidence: Optional[str] = Query(None, description="Filter by confidence: LOW, MEDIUM, HIGH"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """
    List recent trading signals.

    Signals are generated every 15 minutes by the analysis workflow.
    """
    from app.services.database import get_signals

    signals = await get_signals(
        asset=asset,
        direction=direction,
        confidence=confidence,
        limit=limit,
        offset=offset,
    )

    return [
        SignalResponse(
            id=s["id"],
            signal_id=s["signal_id"],
            asset=s["asset"],
            direction=s["direction"],
            confidence=s["confidence"],
            recommended_leverage=s["recommended_leverage"],
            strategy_tag=s["strategy_tag"],
            reasoning=s["reasoning"],
            max_drawdown_30d=s.get("max_drawdown_30d"),
            volatility_score=s.get("volatility_score"),
            generated_at=s["generated_at"],
            expires_at=s["expires_at"],
            is_processed=s["is_processed"],
            users_notified=s.get("users_notified", 0),
            orders_executed=s.get("orders_executed", 0),
        )
        for s in signals
    ]


@router.get("/active", response_model=List[SignalResponse])
async def list_active_signals(
    user: UserInfo = Depends(get_current_user),
):
    """
    List currently active (not expired) signals.
    """
    from app.services.database import get_active_signals

    signals = await get_active_signals()

    return [
        SignalResponse(
            id=s["id"],
            signal_id=s["signal_id"],
            asset=s["asset"],
            direction=s["direction"],
            confidence=s["confidence"],
            recommended_leverage=s["recommended_leverage"],
            strategy_tag=s["strategy_tag"],
            reasoning=s["reasoning"],
            max_drawdown_30d=s.get("max_drawdown_30d"),
            volatility_score=s.get("volatility_score"),
            generated_at=s["generated_at"],
            expires_at=s["expires_at"],
            is_processed=s["is_processed"],
            users_notified=s.get("users_notified", 0),
            orders_executed=s.get("orders_executed", 0),
        )
        for s in signals
    ]


@router.get("/stats", response_model=SignalStatsResponse)
async def get_signal_stats_endpoint(
    user: UserInfo = Depends(get_current_user),
):
    """
    Get signal statistics and performance metrics.
    """
    from app.services.database import get_signal_stats

    stats = await get_signal_stats()

    return SignalStatsResponse(**stats)


@router.get("/{signal_id}", response_model=SignalResponse)
async def get_signal(
    signal_id: str,
    user: UserInfo = Depends(get_current_user),
):
    """
    Get detailed information about a specific signal.
    """
    from app.services.database import get_db
    from sqlalchemy import text as sql_text

    async with get_db() as db:
        query = sql_text("""
            SELECT id, "signalId", direction, asset, confidence,
                   "recommendedLeverage", reasoning, "strategyTag",
                   "maxDrawdown30d", "volatilityScore",
                   "generatedAt", "expiresAt", "isProcessed",
                   "usersNotified", "ordersExecuted"
            FROM agent_signals
            WHERE "signalId" = :signal_id OR id = :signal_id
            LIMIT 1
        """)
        result = await db.execute(query, {"signal_id": signal_id})
        row = result.fetchone()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Signal not found",
        )

    return SignalResponse(
        id=row.id,
        signal_id=row.signalId,
        asset=row.asset,
        direction=row.direction,
        confidence=row.confidence,
        recommended_leverage=row.recommendedLeverage,
        strategy_tag=row.strategyTag,
        reasoning=row.reasoning,
        max_drawdown_30d=row.maxDrawdown30d,
        volatility_score=row.volatilityScore,
        generated_at=row.generatedAt,
        expires_at=row.expiresAt,
        is_processed=row.isProcessed,
        users_notified=row.usersNotified or 0,
        orders_executed=row.ordersExecuted or 0,
    )


@router.get("/{signal_id}/executions")
async def get_signal_executions(
    signal_id: str,
    user: UserInfo = Depends(get_current_user),
):
    """
    Get all executions triggered by a specific signal.
    """
    from app.services.database import get_db
    from sqlalchemy import text

    async with get_db() as db:
        query = text("""
            SELECT id, "missionId", action, asset, quantity, price,
                   success, "hyperliquidTxHash", "executedAt"
            FROM agent_trade_executions
            WHERE "signalId" = :signal_id
            ORDER BY "executedAt" DESC
        """)

        result = await db.execute(query, {"signal_id": signal_id})
        rows = result.fetchall()

        return [
            {
                "id": row.id,
                "mission_id": row.missionId,
                "action": row.action,
                "asset": row.asset,
                "quantity": float(row.quantity) if row.quantity else None,
                "price": float(row.price) if row.price else None,
                "success": row.success,
                "tx_hash": row.hyperliquidTxHash,
                "executed_at": row.executedAt,
            }
            for row in rows
        ]

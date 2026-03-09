"""
LLM Competition Scorecard API

Compares DeepSeek vs Qwen performance across 4 key metrics:
1. Profitability — total yield per provider
2. Drawdown — capital protection during crashes
3. Win Rate — conviction (fewer losing trades)
4. Latency — API response speed
"""

from typing import Optional

import structlog
from fastapi import APIRouter, Depends
from sqlalchemy import text

from app.api.deps import UserInfo, get_current_user
from app.services.database import get_db

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get("/scorecard")
async def get_competition_scorecard(
    user: UserInfo = Depends(get_current_user),
):
    """
    Get the LLM competition scorecard comparing DeepSeek vs Qwen.

    Returns aggregated performance metrics for each LLM provider
    across all missions.
    """
    async with get_db() as db:
        # 1. Mission-level performance by provider
        mission_stats = await db.execute(text("""
            SELECT
                COALESCE(m."llmProvider", 'deepseek') as provider,
                COUNT(*) as total_missions,
                COUNT(*) FILTER (WHERE m.status = 'ACTIVE') as active_missions,
                COUNT(*) FILTER (WHERE m.status = 'COMPLETED') as completed_missions,
                COALESCE(SUM(m."totalPnl"), 0) as total_pnl,
                COALESCE(AVG(m."totalPnl"), 0) as avg_pnl,
                COALESCE(AVG(m."winRate"), 0) as avg_win_rate,
                COALESCE(AVG(m."maxDrawdown"), 0) as avg_max_drawdown,
                MAX(m."maxDrawdown") as worst_drawdown,
                COALESCE(SUM(m."totalTrades"), 0) as total_trades,
                COALESCE(AVG(CASE
                    WHEN m."initialCapital" > 0 THEN
                        (m."totalPnl" / m."initialCapital" * 100)
                    ELSE 0
                END), 0) as avg_return_pct
            FROM agent_missions m
            WHERE m.status IN ('ACTIVE', 'COMPLETED', 'PAUSED')
            GROUP BY COALESCE(m."llmProvider", 'deepseek')
        """))
        mission_rows = mission_stats.fetchall()

        # 2. Signal-level performance by provider (latency + conviction)
        signal_stats = await db.execute(text("""
            SELECT
                COALESCE(s."llmProvider", 'deepseek') as provider,
                COUNT(*) as total_signals,
                AVG(s."responseTimeMs") as avg_response_time_ms,
                MIN(s."responseTimeMs") as min_response_time_ms,
                MAX(s."responseTimeMs") as max_response_time_ms,
                COUNT(*) FILTER (WHERE s.confidence = 'HIGH') as high_confidence_count,
                COUNT(*) FILTER (WHERE s.confidence = 'MEDIUM') as medium_confidence_count,
                COUNT(*) FILTER (WHERE s.confidence = 'LOW') as low_confidence_count
            FROM agent_signals s
            GROUP BY COALESCE(s."llmProvider", 'deepseek')
        """))
        signal_rows = signal_stats.fetchall()

        # 3. Position-level win/loss by provider
        position_stats = await db.execute(text("""
            SELECT
                COALESCE(m."llmProvider", 'deepseek') as provider,
                COUNT(*) as total_positions,
                COUNT(*) FILTER (WHERE p."realizedPnl" > 0) as winning_positions,
                COUNT(*) FILTER (WHERE p."realizedPnl" <= 0) as losing_positions,
                COALESCE(AVG(p."realizedPnl"), 0) as avg_pnl_per_position,
                COALESCE(SUM(p."realizedPnl"), 0) as total_realized_pnl,
                COALESCE(AVG(CASE WHEN p."realizedPnl" > 0 THEN p."realizedPnl" ELSE NULL END), 0) as avg_win,
                COALESCE(AVG(CASE WHEN p."realizedPnl" <= 0 THEN p."realizedPnl" ELSE NULL END), 0) as avg_loss
            FROM agent_positions p
            JOIN agent_missions m ON p."missionId" = m.id
            WHERE p.status = 'CLOSED'
            GROUP BY COALESCE(m."llmProvider", 'deepseek')
        """))
        position_rows = position_stats.fetchall()

    # Build scorecard
    providers = {}
    for row in mission_rows:
        providers[row.provider] = {
            "provider": row.provider,
            "missions": {
                "total": row.total_missions,
                "active": row.active_missions,
                "completed": row.completed_missions,
            },
            "profitability": {
                "total_pnl": float(row.total_pnl),
                "avg_pnl_per_mission": float(row.avg_pnl),
                "avg_return_pct": round(float(row.avg_return_pct), 2),
                "total_trades": row.total_trades,
            },
            "drawdown": {
                "avg_max_drawdown_pct": round(float(row.avg_max_drawdown), 2),
                "worst_drawdown_pct": round(float(row.worst_drawdown or 0), 2),
            },
            "conviction": {
                "avg_win_rate_pct": round(float(row.avg_win_rate), 2),
            },
        }

    for row in signal_rows:
        if row.provider in providers:
            providers[row.provider]["latency"] = {
                "avg_response_time_ms": round(float(row.avg_response_time_ms or 0)),
                "min_response_time_ms": row.min_response_time_ms,
                "max_response_time_ms": row.max_response_time_ms,
                "total_signals_generated": row.total_signals,
            }
            providers[row.provider]["conviction"]["high_confidence_signals"] = row.high_confidence_count
            providers[row.provider]["conviction"]["medium_confidence_signals"] = row.medium_confidence_count
            providers[row.provider]["conviction"]["low_confidence_signals"] = row.low_confidence_count

    for row in position_rows:
        if row.provider in providers:
            total = row.total_positions or 1
            providers[row.provider]["conviction"]["winning_positions"] = row.winning_positions
            providers[row.provider]["conviction"]["losing_positions"] = row.losing_positions
            providers[row.provider]["conviction"]["win_rate_from_positions_pct"] = round(
                row.winning_positions / total * 100, 2
            )
            providers[row.provider]["profitability"]["total_realized_pnl"] = float(row.total_realized_pnl)
            providers[row.provider]["profitability"]["avg_pnl_per_position"] = round(float(row.avg_pnl_per_position), 4)
            providers[row.provider]["profitability"]["avg_win_size"] = round(float(row.avg_win), 4)
            providers[row.provider]["profitability"]["avg_loss_size"] = round(float(row.avg_loss), 4)

    # Determine leader for each metric
    provider_list = list(providers.values())
    verdict = {}
    if len(provider_list) == 2:
        a, b = provider_list
        verdict["profitability_leader"] = (
            a["provider"] if a["profitability"]["avg_return_pct"] > b["profitability"]["avg_return_pct"]
            else b["provider"]
        )
        verdict["drawdown_leader"] = (
            a["provider"] if a["drawdown"]["avg_max_drawdown_pct"] < b["drawdown"]["avg_max_drawdown_pct"]
            else b["provider"]
        )
        verdict["conviction_leader"] = (
            a["provider"] if a["conviction"]["avg_win_rate_pct"] > b["conviction"]["avg_win_rate_pct"]
            else b["provider"]
        )
        if a.get("latency") and b.get("latency"):
            verdict["latency_leader"] = (
                a["provider"] if a["latency"]["avg_response_time_ms"] < b["latency"]["avg_response_time_ms"]
                else b["provider"]
            )

    return {
        "scorecard": provider_list,
        "verdict": verdict,
        "competition_status": "active" if len(provider_list) == 2 else "single_provider",
    }

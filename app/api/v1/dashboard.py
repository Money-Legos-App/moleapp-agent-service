"""
Dashboard API Endpoints
Aggregated data for the mobile DeFi dashboard
"""

from datetime import datetime
from typing import List, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict

from app.api.deps import UserInfo, get_current_user

logger = structlog.get_logger(__name__)
router = APIRouter()


# ==================
# Camel case serialization
# ==================

def _to_camel(string: str) -> str:
    parts = string.split("_")
    return parts[0] + "".join(w.capitalize() for w in parts[1:])


# ==================
# Response Models
# ==================

class FundingInfo(BaseModel):
    """Funding information."""
    model_config = ConfigDict(alias_generator=_to_camel, populate_by_name=True)

    initial_capital: float
    current_value: float
    currency: str = "USDC"


class PerformanceInfo(BaseModel):
    """Performance metrics."""
    model_config = ConfigDict(alias_generator=_to_camel, populate_by_name=True)

    total_pnl: float
    total_pnl_percent: float
    unrealized_pnl: float
    realized_pnl: float
    win_rate: float
    total_trades: int
    max_drawdown: float


class PositionInfo(BaseModel):
    """Position summary for dashboard."""
    model_config = ConfigDict(alias_generator=_to_camel, populate_by_name=True)

    asset: str
    direction: str
    size: float
    entry_price: float
    current_price: float
    leverage: int
    unrealized_pnl: float
    unrealized_pnl_percent: float


class RecentTradeInfo(BaseModel):
    """Recent trade information."""
    model_config = ConfigDict(alias_generator=_to_camel, populate_by_name=True)

    asset: str
    action: str
    pnl: float
    executed_at: datetime


class LiveBalanceInfo(BaseModel):
    """Live on-chain balance from Hyperliquid clearinghouse."""
    model_config = ConfigDict(alias_generator=_to_camel, populate_by_name=True)

    account_value: Optional[float] = None
    withdrawable: Optional[float] = None
    margin_used: Optional[float] = None
    is_live: bool = False


class MissionDashboard(BaseModel):
    """Complete mission dashboard data."""
    model_config = ConfigDict(alias_generator=_to_camel, populate_by_name=True)

    mission_id: str
    status: str
    strategy: str
    day_number: int
    days_remaining: int
    funding: FundingInfo
    performance: PerformanceInfo
    positions: List[PositionInfo]
    recent_trades: List[RecentTradeInfo]
    live_balance: Optional[LiveBalanceInfo] = None


class PnLDataPoint(BaseModel):
    """PnL history data point."""
    model_config = ConfigDict(alias_generator=_to_camel, populate_by_name=True)

    timestamp: datetime
    total_value: float
    pnl: float


class PnLHistoryResponse(BaseModel):
    """PnL history for charts."""
    model_config = ConfigDict(alias_generator=_to_camel, populate_by_name=True)

    mission_id: str
    data_points: List[PnLDataPoint]


class TradeRecord(BaseModel):
    """Trade execution record."""
    model_config = ConfigDict(alias_generator=_to_camel, populate_by_name=True)

    id: str
    signal_id: Optional[str]
    asset: str
    direction: str
    action: str
    quantity: float
    price: float
    leverage: int
    pnl: Optional[float]
    mission_day: int
    reasoning: str
    executed_at: datetime


class TradesResponse(BaseModel):
    """Trade history response."""
    model_config = ConfigDict(alias_generator=_to_camel, populate_by_name=True)

    mission_id: str
    trades: List[TradeRecord]
    total_count: int


# ==================
# Helpers
# ==================

async def _fetch_live_hl_balance(master_eoa_address: str) -> Optional[dict]:
    """Fetch live balance from HL. Returns None on any failure (never raises)."""
    if not master_eoa_address:
        return None
    try:
        import asyncio
        from app.services.hyperliquid import HyperliquidClient
        hl_client = HyperliquidClient()
        try:
            state = await asyncio.wait_for(
                hl_client.get_clearinghouse_state(master_eoa_address),
                timeout=5.0,
            )
            return state.get("account")
        finally:
            await hl_client.close()
    except Exception as e:
        logger.warning("Live HL balance fetch failed (using DB fallback)", error=str(e))
        return None


# ==================
# Endpoints
# ==================

@router.get("/missions/{mission_id}/summary", response_model=MissionDashboard)
async def get_mission_dashboard(
    mission_id: str,
    user: UserInfo = Depends(get_current_user),
):
    """
    Get complete mission dashboard data.

    This is the main endpoint for the mobile DeFi dashboard.
    Includes:
    - Mission status and progress
    - Funding and current value
    - Performance metrics
    - Current positions with live prices
    - Recent trades
    - Live on-chain balance from Hyperliquid
    """
    from app.services.database import (
        get_mission_by_id,
        get_open_positions,
        get_trade_executions,
    )

    logger.info("Getting mission dashboard", mission_id=mission_id, user_id=user.user_id)

    mission = await get_mission_by_id(mission_id, user.user_id)
    if not mission:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Mission not found",
        )

    # Fetch open positions, recent trades, and live HL balance concurrently
    import asyncio
    positions_task = get_open_positions(mission_id)
    trades_task = get_trade_executions(mission_id, user.user_id, limit=5, offset=0)
    hl_task = _fetch_live_hl_balance(mission.get("master_eoa_address", ""))

    positions, (trades_raw, _), live_account = await asyncio.gather(
        positions_task, trades_task, hl_task
    )

    initial_capital = mission["initial_capital"]

    # Prefer live HL account value when available
    live_balance = None
    if live_account:
        live_balance = LiveBalanceInfo(
            account_value=live_account["account_value"],
            withdrawable=live_account["withdrawable"],
            margin_used=live_account["total_margin_used"],
            is_live=True,
        )
        current_value = live_account["account_value"]
    else:
        current_value = mission["current_value"] if mission["current_value"] else initial_capital

    total_pnl = mission["total_pnl"]

    # Sum unrealized from open positions
    total_unrealized = sum(p.get("unrealized_pnl", 0) for p in positions)
    realized = total_pnl - total_unrealized

    return MissionDashboard(
        mission_id=mission["id"],
        status=mission["status"],
        strategy=mission["strategy"],
        day_number=mission["day_number"],
        days_remaining=mission["days_remaining"] or 0,
        funding=FundingInfo(
            initial_capital=initial_capital,
            current_value=current_value,
        ),
        performance=PerformanceInfo(
            total_pnl=total_pnl,
            total_pnl_percent=mission["total_pnl_percent"],
            unrealized_pnl=total_unrealized,
            realized_pnl=realized,
            win_rate=mission["win_rate"],
            total_trades=mission["total_trades"],
            max_drawdown=mission["max_drawdown"],
        ),
        positions=[
            PositionInfo(
                asset=p["asset"],
                direction=p["direction"],
                size=p["quantity"],
                entry_price=p["entry_price"],
                current_price=p["current_price"],
                leverage=p["leverage"],
                unrealized_pnl=p["unrealized_pnl"],
                unrealized_pnl_percent=(
                    (p["unrealized_pnl"] / (p["entry_price"] * p["quantity"]) * 100)
                    if p["entry_price"] * p["quantity"] > 0
                    else 0
                ),
            )
            for p in positions
        ],
        recent_trades=[
            RecentTradeInfo(
                asset=t["asset"],
                action=t["action"],
                pnl=0,  # PnL per trade not tracked at execution level
                executed_at=t["executed_at"],
            )
            for t in trades_raw
        ],
        live_balance=live_balance,
    )


# ==================
# Activity Feed (AI Reasoning Transparency)
# ==================

class ActivityEntry(BaseModel):
    """A single activity entry in the neural feed."""
    model_config = ConfigDict(alias_generator=_to_camel, populate_by_name=True)

    id: str
    timestamp: datetime
    node: str
    action: str
    activity_type: str  # scan, recall, act, heartbeat
    asset: Optional[str] = None
    reasoning: Optional[str] = None
    decision_summary: Optional[str] = None
    success: bool = True
    llm_provider: Optional[str] = None


class ActivityFeedResponse(BaseModel):
    """Neural Activity Feed response."""
    model_config = ConfigDict(alias_generator=_to_camel, populate_by_name=True)

    mission_id: str
    activities: List[ActivityEntry]
    has_more: bool


def _map_node_to_activity_type(node: str, action: str) -> str:
    """Map audit log node/action to mobile ActivityType."""
    if node == "market_analysis":
        return "scan"
    if node == "user_filter":
        return "recall"
    if node in ("execution", "trade", "lifecycle"):
        return "act"
    return "heartbeat"


def _format_node_message(node: str, action: str, asset: Optional[str] = None) -> str:
    """Human-readable message from audit log node/action."""
    asset_label = f" on {asset}" if asset else ""
    mapping = {
        "market_analysis": f"Market analysis{asset_label}",
        "user_filter": f"Strategy filter applied{asset_label}",
        "execution": f"Trade {action.replace('_', ' ')}{asset_label}",
        "monitoring": f"Risk monitoring check{asset_label}",
        "lifecycle": f"Mission {action.replace('_', ' ')}",
    }
    return mapping.get(node, f"Agent {action.replace('_', ' ')}{asset_label}")


@router.get("/missions/{mission_id}/activity", response_model=ActivityFeedResponse)
async def get_mission_activity(
    mission_id: str,
    user: UserInfo = Depends(get_current_user),
    limit: int = Query(20, ge=1, le=50),
):
    """
    Neural Activity Feed: recent AI decisions and reasoning for a mission.
    Surfaces audit log data without exposing raw LLM prompts.
    """
    from app.services.database import get_mission_by_id, get_audit_logs_by_mission

    mission = await get_mission_by_id(mission_id, user.user_id)
    if not mission:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Mission not found",
        )

    audit_logs = await get_audit_logs_by_mission(mission_id, limit=limit)

    activities = []
    for log in audit_logs:
        activity_type = _map_node_to_activity_type(log["node"], log["action"])

        # Extract decision summary from JSON
        decision_summary = None
        if log.get("decision") and isinstance(log["decision"], dict):
            d = log["decision"]
            decision_summary = d.get("summary") or d.get("action") or d.get("direction")

        activities.append(ActivityEntry(
            id=log["id"],
            timestamp=log["created_at"],
            node=log["node"],
            action=log["action"],
            activity_type=activity_type,
            asset=log.get("asset"),
            reasoning=log.get("reasoning"),
            decision_summary=decision_summary,
            success=log.get("success", True),
            llm_provider=log.get("llm_model"),
        ))

    return ActivityFeedResponse(
        mission_id=mission_id,
        activities=activities,
        has_more=len(audit_logs) >= limit,
    )


@router.get("/missions/{mission_id}/pnl-history", response_model=PnLHistoryResponse)
async def get_pnl_history(
    mission_id: str,
    user: UserInfo = Depends(get_current_user),
    days: int = Query(30, ge=1, le=90, description="Number of days of history"),
):
    """
    Get PnL history for charts.

    Returns daily snapshots of total value and PnL.
    """
    logger.info(
        "Getting PnL history",
        mission_id=mission_id,
        user_id=user.user_id,
        days=days,
    )

    from app.services.database import get_pnl_snapshots, get_mission_by_id

    # Verify ownership
    mission = await get_mission_by_id(mission_id, user.user_id)
    if not mission:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Mission not found",
        )

    snapshots = await get_pnl_snapshots(mission_id, days)

    return PnLHistoryResponse(
        mission_id=mission_id,
        data_points=[
            PnLDataPoint(
                timestamp=s["timestamp"],
                total_value=s["total_value"],
                pnl=s["total_pnl"],
            )
            for s in snapshots
        ],
    )


@router.get("/missions/{mission_id}/trades", response_model=TradesResponse)
async def get_trade_history(
    mission_id: str,
    user: UserInfo = Depends(get_current_user),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """
    Get trade history for a mission.

    Includes all executed trades with reasoning from the AI.
    """
    logger.info(
        "Getting trade history",
        mission_id=mission_id,
        user_id=user.user_id,
        limit=limit,
    )

    from app.services.database import get_trade_executions, get_mission_by_id

    # Verify ownership
    mission = await get_mission_by_id(mission_id, user.user_id)
    if not mission:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Mission not found",
        )

    trades_raw, total_count = await get_trade_executions(
        mission_id, user.user_id, limit, offset
    )

    return TradesResponse(
        mission_id=mission_id,
        trades=[
            TradeRecord(
                id=t["id"],
                signal_id=t.get("signal_id"),
                asset=t["asset"],
                direction="LONG" if t["action"] in ("ENTER_LONG", "EXIT_SHORT") else "SHORT",
                action=t["action"],
                quantity=t["quantity"],
                price=t["price"],
                leverage=1,  # leverage not stored per-trade; from mission config
                pnl=None,
                mission_day=t["mission_day"],
                reasoning=t["decision_reason"],
                executed_at=t["executed_at"],
            )
            for t in trades_raw
        ],
        total_count=total_count,
    )


@router.get("/overview")
async def get_user_overview(
    user: UserInfo = Depends(get_current_user),
):
    """
    Get overview of all user's agent missions.

    Aggregates data across all missions for a high-level view.
    """
    logger.info("Getting user overview", user_id=user.user_id)

    from app.services.database import get_user_missions_aggregate, get_missions_by_user

    agg = await get_user_missions_aggregate(user.user_id)
    missions = await get_missions_by_user(user.user_id)

    return {
        "userId": user.user_id,
        "totalMissions": agg.get("total_missions", 0),
        "activeMissions": agg.get("active_missions", 0),
        "totalCapitalDeployed": agg.get("total_capital_deployed", 0),
        "totalPnl": agg.get("total_pnl", 0),
        "totalPnlPercent": agg.get("total_pnl_percent", 0),
        "missions": missions,
    }


@router.get("/live-prices")
async def get_live_prices(
    assets: Optional[str] = Query(None, description="Comma-separated list of assets"),
):
    """
    Get live prices for assets.

    Used for real-time dashboard updates.
    """
    from app.services.hyperliquid import HyperliquidClient

    hl_client = HyperliquidClient()

    try:
        if assets:
            asset_list = [a.strip() for a in assets.split(",")]
        else:
            from app.config import get_settings
            asset_list = get_settings().allowed_assets

        prices = await hl_client.get_mark_prices(asset_list)

        return {
            "prices": {
                asset: {
                    "price": price,
                    "asset": asset,
                }
                for asset, price in prices.items()
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

    finally:
        await hl_client.close()

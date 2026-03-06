"""
Positions API Endpoints
View and manage trading positions
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

class PositionResponse(BaseModel):
    """Trading position response."""
    model_config = ConfigDict(alias_generator=_to_camel, populate_by_name=True)

    id: str
    mission_id: str
    asset: str
    direction: str  # LONG or SHORT
    entry_price: str
    current_price: str
    quantity: str
    leverage: int
    margin_used: str
    liquidation_price: Optional[str]
    unrealized_pnl: str
    unrealized_pnl_percent: float
    realized_pnl: str
    funding_paid: str
    status: str  # OPEN, CLOSED, LIQUIDATED
    stop_loss_price: Optional[str]
    take_profit_price: Optional[str]
    opened_at: datetime
    closed_at: Optional[datetime]
    close_reason: Optional[str]


class PositionSummaryResponse(BaseModel):
    """Position summary for dashboard."""
    model_config = ConfigDict(alias_generator=_to_camel, populate_by_name=True)

    asset: str
    direction: str
    size_usd: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_percent: float
    leverage: int


class PortfolioResponse(BaseModel):
    """Portfolio overview."""
    model_config = ConfigDict(alias_generator=_to_camel, populate_by_name=True)

    total_value: float
    total_margin_used: float
    available_margin: float
    total_unrealized_pnl: float
    total_realized_pnl: float
    open_positions_count: int
    positions: List[PositionSummaryResponse]


# ==================
# Endpoints
# ==================

@router.get("", response_model=List[PositionResponse])
async def list_positions(
    user: UserInfo = Depends(get_current_user),
    mission_id: Optional[str] = Query(None, description="Filter by mission"),
    asset: Optional[str] = Query(None, description="Filter by asset"),
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status: OPEN, CLOSED, LIQUIDATED"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """
    List positions for the current user.

    Optionally filter by mission, asset, or status.
    """
    from app.services.database import get_positions_by_user

    logger.info(
        "Listing positions",
        user_id=user.user_id,
        mission_id=mission_id,
        asset=asset,
        status=status_filter,
    )

    positions = await get_positions_by_user(
        user.user_id,
        mission_id=mission_id,
        status_filter=status_filter,
        limit=limit,
        offset=offset,
    )

    return [
        PositionResponse(
            id=p["id"],
            mission_id=p["mission_id"],
            asset=p["asset"],
            direction=p["direction"],
            entry_price=str(p["entry_price"]),
            current_price=str(p["current_price"]),
            quantity=str(p["quantity"]),
            leverage=p["leverage"],
            margin_used=str(p["margin_used"]),
            liquidation_price=str(p["liquidation_price"]) if p["liquidation_price"] else None,
            unrealized_pnl=str(p["unrealized_pnl"]),
            unrealized_pnl_percent=(
                (p["unrealized_pnl"] / (p["entry_price"] * p["quantity"]) * 100)
                if p["entry_price"] * p["quantity"] > 0
                else 0
            ),
            realized_pnl=str(p["realized_pnl"]),
            funding_paid=str(p["funding_paid"]),
            status=p["status"],
            stop_loss_price=None,
            take_profit_price=None,
            opened_at=p["opened_at"],
            closed_at=p["closed_at"],
            close_reason=p["close_reason"],
        )
        for p in positions
    ]


@router.get("/open", response_model=List[PositionSummaryResponse])
async def list_open_positions(
    user: UserInfo = Depends(get_current_user),
    mission_id: Optional[str] = Query(None, description="Filter by mission"),
):
    """
    List currently open positions.
    """
    from app.services.database import get_positions_by_user

    logger.info(
        "Listing open positions",
        user_id=user.user_id,
        mission_id=mission_id,
    )

    positions = await get_positions_by_user(
        user.user_id,
        mission_id=mission_id,
        status_filter="OPEN",
    )

    return [
        PositionSummaryResponse(
            asset=p["asset"],
            direction=p["direction"],
            size_usd=p["entry_price"] * p["quantity"],
            entry_price=p["entry_price"],
            current_price=p["current_price"],
            unrealized_pnl=p["unrealized_pnl"],
            unrealized_pnl_percent=(
                (p["unrealized_pnl"] / (p["entry_price"] * p["quantity"]) * 100)
                if p["entry_price"] * p["quantity"] > 0
                else 0
            ),
            leverage=p["leverage"],
        )
        for p in positions
    ]


@router.get("/portfolio", response_model=PortfolioResponse)
async def get_portfolio(
    user: UserInfo = Depends(get_current_user),
    mission_id: Optional[str] = Query(None, description="Filter by mission"),
):
    """
    Get portfolio overview with all open positions.
    """
    from app.services.hyperliquid import HyperliquidClient

    logger.info(
        "Getting portfolio",
        user_id=user.user_id,
        mission_id=mission_id,
    )

    from app.services.database import get_positions_by_user

    positions = await get_positions_by_user(
        user.user_id,
        mission_id=mission_id,
        status_filter="OPEN",
    )

    total_margin = sum(p["margin_used"] for p in positions)
    total_unrealized = sum(p["unrealized_pnl"] for p in positions)
    total_realized = sum(p["realized_pnl"] for p in positions)
    total_value = total_margin + total_unrealized

    return PortfolioResponse(
        total_value=total_value,
        total_margin_used=total_margin,
        available_margin=0,  # Would need mission capital - margin used
        total_unrealized_pnl=total_unrealized,
        total_realized_pnl=total_realized,
        open_positions_count=len(positions),
        positions=[
            PositionSummaryResponse(
                asset=p["asset"],
                direction=p["direction"],
                size_usd=p["entry_price"] * p["quantity"],
                entry_price=p["entry_price"],
                current_price=p["current_price"],
                unrealized_pnl=p["unrealized_pnl"],
                unrealized_pnl_percent=(
                    (p["unrealized_pnl"] / (p["entry_price"] * p["quantity"]) * 100)
                    if p["entry_price"] * p["quantity"] > 0
                    else 0
                ),
                leverage=p["leverage"],
            )
            for p in positions
        ],
    )


@router.get("/{position_id}", response_model=PositionResponse)
async def get_position(
    position_id: str,
    user: UserInfo = Depends(get_current_user),
):
    """
    Get detailed position information.
    """
    from app.services.database import get_positions_by_user

    logger.info("Getting position", position_id=position_id, user_id=user.user_id)

    # Fetch all user positions and find the one by ID
    all_positions = await get_positions_by_user(user.user_id, limit=500)
    position = next((p for p in all_positions if p["id"] == position_id), None)

    if not position:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Position not found",
        )

    p = position
    return PositionResponse(
        id=p["id"],
        mission_id=p["mission_id"],
        asset=p["asset"],
        direction=p["direction"],
        entry_price=str(p["entry_price"]),
        current_price=str(p["current_price"]),
        quantity=str(p["quantity"]),
        leverage=p["leverage"],
        margin_used=str(p["margin_used"]),
        liquidation_price=str(p["liquidation_price"]) if p["liquidation_price"] else None,
        unrealized_pnl=str(p["unrealized_pnl"]),
        unrealized_pnl_percent=(
            (p["unrealized_pnl"] / (p["entry_price"] * p["quantity"]) * 100)
            if p["entry_price"] * p["quantity"] > 0
            else 0
        ),
        realized_pnl=str(p["realized_pnl"]),
        funding_paid=str(p["funding_paid"]),
        status=p["status"],
        stop_loss_price=None,
        take_profit_price=None,
        opened_at=p["opened_at"],
        closed_at=p["closed_at"],
        close_reason=p["close_reason"],
    )


@router.get("/{position_id}/history")
async def get_position_history(
    position_id: str,
    user: UserInfo = Depends(get_current_user),
):
    """
    Get price history and events for a position.
    """
    logger.info("Getting position history", position_id=position_id, user_id=user.user_id)

    # TODO: Implement
    return {
        "position_id": position_id,
        "events": [],
        "price_history": [],
    }

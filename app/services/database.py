"""
Database Service
SQLAlchemy async database operations for the agent-service
"""

from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, AsyncGenerator, Dict, List, Optional

import structlog
from sqlalchemy import and_, select, update, func, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import Column, String, Boolean, Integer, Float, DateTime, ARRAY, ForeignKey, Enum, Numeric, Text
from sqlalchemy.dialects.postgresql import JSONB

from app.config import get_settings

logger = structlog.get_logger(__name__)

# SQLAlchemy Base
Base = declarative_base()

# Engine and session factory (lazy initialization)
_engine = None
_session_factory = None


def get_engine():
    """Get or create the async engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        # Convert postgresql:// to postgresql+asyncpg://
        db_url = settings.database_url.replace(
            "postgresql://", "postgresql+asyncpg://"
        )
        _engine = create_async_engine(
            db_url,
            echo=settings.debug,
            pool_size=5,
            max_overflow=10,
        )
    return _engine


def get_session_factory():
    """Get or create the session factory."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


@asynccontextmanager
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get a database session as an async context manager."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


# ==================
# Database Models (SQLAlchemy mirror of Prisma schema)
# ==================

class AgentMission(Base):
    """Agent mission model."""
    __tablename__ = "agent_missions"

    id = Column(String, primary_key=True)
    userId = Column(String, nullable=False)
    walletId = Column(String, nullable=False)
    turnkeySignerId = Column(String, nullable=True)
    hyperliquidApproved = Column(Boolean, default=False)
    approvalTxHash = Column(String, nullable=True)
    strategy = Column(String, default="SHORT_TERM_30D")
    riskLevel = Column(String, default="MODERATE")
    durationDays = Column(Integer, default=30)
    startedAt = Column(DateTime, nullable=True)
    endsAt = Column(DateTime, nullable=True)
    initialCapital = Column(Numeric(20, 8), nullable=False)
    currentValue = Column(Numeric(20, 8), nullable=True)
    depositCurrency = Column(String, default="USDC")
    maxLeverage = Column(Integer, default=3)
    allowedAssets = Column(ARRAY(String), default=["ETH-USD", "BTC-USD"])
    sessionKeyId = Column(String, nullable=True)
    status = Column(String, default="PENDING")
    pausedAt = Column(DateTime, nullable=True)
    completedAt = Column(DateTime, nullable=True)
    totalPnl = Column(Numeric(20, 8), default=0)
    totalTrades = Column(Integer, default=0)
    winRate = Column(Float, default=0)
    maxDrawdown = Column(Float, default=0)
    createdAt = Column(DateTime, default=datetime.utcnow)
    updatedAt = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TurnkeySigner(Base):
    """Turnkey signer model (read-only, managed by wallet-service)."""
    __tablename__ = "turnkey_signers"

    id = Column(String, primary_key=True)
    userId = Column(String, nullable=False)
    walletId = Column(String, nullable=True)
    turnkeySubOrgId = Column(String, nullable=False)
    turnkeyUserId = Column(String, nullable=True)
    address = Column(String, nullable=True)
    isActive = Column(Boolean, default=True)


class AgentPosition(Base):
    """Agent position model."""
    __tablename__ = "agent_positions"

    id = Column(String, primary_key=True)
    missionId = Column(String, nullable=False)
    signalId = Column(String, nullable=True)
    asset = Column(String, nullable=False)
    direction = Column(String, nullable=False)
    entryPrice = Column(Numeric(20, 8), nullable=False)
    currentPrice = Column(Numeric(20, 8), nullable=False)
    quantity = Column(Numeric(20, 8), nullable=False)
    leverage = Column(Integer, default=1)
    hyperliquidOrderId = Column(String, nullable=True)
    marginUsed = Column(Numeric(20, 8), nullable=False)
    liquidationPrice = Column(Numeric(20, 8), nullable=True)
    unrealizedPnl = Column(Numeric(20, 8), default=0)
    realizedPnl = Column(Numeric(20, 8), default=0)
    fundingPaid = Column(Numeric(20, 8), default=0)
    stopLossPrice = Column(Numeric(20, 8), nullable=True)
    takeProfitPrice = Column(Numeric(20, 8), nullable=True)
    status = Column(String, default="OPEN")
    openedAt = Column(DateTime, default=datetime.utcnow)
    closedAt = Column(DateTime, nullable=True)
    closeReason = Column(String, nullable=True)


class AgentTradeExecution(Base):
    """Agent trade execution model."""
    __tablename__ = "agent_trade_executions"

    id = Column(String, primary_key=True)
    missionId = Column(String, nullable=False)
    signalId = Column(String, nullable=True)
    action = Column(String, nullable=False)
    asset = Column(String, nullable=False)
    quantity = Column(Numeric(20, 8), nullable=True)
    price = Column(Numeric(20, 8), nullable=True)
    missionDay = Column(Integer, nullable=False)
    decisionReason = Column(String, nullable=False)
    userBalanceAtTime = Column(Numeric(20, 8), nullable=False)
    success = Column(Boolean, nullable=False)
    hyperliquidTxHash = Column(String, nullable=True)
    errorMessage = Column(String, nullable=True)
    executedAt = Column(DateTime, default=datetime.utcnow)


class AgentPnLSnapshot(Base):
    """Agent PnL snapshot model."""
    __tablename__ = "agent_pnl_snapshots"

    id = Column(String, primary_key=True)
    missionId = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    totalValue = Column(Numeric(20, 8), nullable=False)
    totalPnl = Column(Numeric(20, 8), nullable=False)
    unrealizedPnl = Column(Numeric(20, 8), nullable=False)
    realizedPnl = Column(Numeric(20, 8), nullable=False)


class AgentAuditLog(Base):
    """Agent audit log for tracking all LLM decisions, trade events, and lifecycle actions."""
    __tablename__ = "agent_audit_logs"

    id = Column(String, primary_key=True)
    missionId = Column(String, nullable=True)
    userId = Column(String, nullable=True)
    node = Column(String, nullable=False)       # market_analysis, user_filter, execution, monitoring, lifecycle
    action = Column(String, nullable=False)      # llm_call, signal_generated, signal_skipped, filter_decision, etc.
    asset = Column(String, nullable=True)

    # LLM audit fields
    llmPrompt = Column(Text, nullable=True)
    llmResponse = Column(Text, nullable=True)
    llmModel = Column(String, nullable=True)
    llmTokens = Column(Integer, nullable=True)

    # Decision fields
    decision = Column(JSONB, nullable=True)
    reasoning = Column(Text, nullable=True)

    # Context
    meta = Column("metadata", JSONB, nullable=True)
    success = Column(Boolean, default=True)
    errorMessage = Column(Text, nullable=True)
    dryRun = Column(Boolean, default=False)

    createdAt = Column(DateTime, default=datetime.utcnow)


class AgentSignal(Base):
    """Agent signal model — persisted trading signals for backtesting and attribution."""
    __tablename__ = "agent_signals"

    id = Column(String, primary_key=True)
    signalId = Column(String, unique=True, nullable=False)
    direction = Column(String, nullable=False)
    asset = Column(String, nullable=False)
    confidence = Column(String, nullable=False)
    recommendedLeverage = Column(Integer, default=1)
    reasoning = Column(Text, nullable=False)
    strategyTag = Column(String, nullable=False)
    ragContextIds = Column(ARRAY(String), default=[])
    maxDrawdown30d = Column(Float, nullable=True)
    volatilityScore = Column(Float, nullable=True)
    generatedAt = Column(DateTime, default=datetime.utcnow)
    expiresAt = Column(DateTime, nullable=False)
    isProcessed = Column(Boolean, default=False)
    processedAt = Column(DateTime, nullable=True)
    usersNotified = Column(Integer, default=0)
    ordersGenerated = Column(Integer, default=0)
    ordersExecuted = Column(Integer, default=0)


# ==================
# Query Functions
# ==================

async def get_active_missions() -> List[Dict[str, Any]]:
    """
    Fetch all active missions with their Turnkey signer details.

    Returns:
        List of mission dictionaries with wallet information
    """
    async with get_db() as db:
        # Use raw SQL for the JOIN since models might not have relationships defined
        query = text("""
            SELECT
                m.id,
                m."userId" as user_id,
                m."walletId" as wallet_id,
                m."turnkeySignerId" as turnkey_signer_id,
                m."hyperliquidApproved" as hyperliquid_approved,
                m.strategy,
                m."riskLevel" as risk_level,
                m."durationDays" as duration_days,
                m."startedAt" as started_at,
                m."endsAt" as ends_at,
                m."initialCapital" as initial_capital,
                m."currentValue" as current_value,
                m."maxLeverage" as max_leverage,
                m."allowedAssets" as allowed_assets,
                m.status,
                m."totalPnl" as total_pnl,
                m."totalTrades" as total_trades,
                m."winRate" as win_rate,
                m."maxDrawdown" as max_drawdown,
                t."turnkeySubOrgId" as turnkey_sub_org_id,
                t."turnkeyUserId" as turnkey_user_id,
                t.address as user_wallet_address
            FROM agent_missions m
            LEFT JOIN turnkey_signers t ON m."turnkeySignerId" = t.id
            WHERE m.status = 'ACTIVE'
            AND m."hyperliquidApproved" = true
            ORDER BY m."startedAt" ASC
        """)

        result = await db.execute(query)
        rows = result.fetchall()

        missions = []
        for row in rows:
            missions.append({
                "id": row.id,
                "user_id": row.user_id,
                "wallet_id": row.wallet_id,
                "turnkey_signer_id": row.turnkey_signer_id,
                "hyperliquid_approved": row.hyperliquid_approved,
                "strategy": row.strategy,
                "risk_level": row.risk_level,
                "duration_days": row.duration_days,
                "started_at": row.started_at,
                "ends_at": row.ends_at,
                "initial_capital": float(row.initial_capital) if row.initial_capital else 0,
                "current_value": float(row.current_value) if row.current_value else None,
                "max_leverage": row.max_leverage,
                "allowed_assets": row.allowed_assets or ["ETH-USD", "BTC-USD"],
                "status": row.status,
                "total_pnl": float(row.total_pnl) if row.total_pnl else 0,
                "total_trades": row.total_trades or 0,
                "win_rate": row.win_rate or 0,
                "max_drawdown": row.max_drawdown or 0,
                "turnkey_sub_org_id": row.turnkey_sub_org_id,
                "turnkey_user_id": row.turnkey_user_id,
                "user_wallet_address": row.user_wallet_address,
            })

        logger.info("Fetched active missions", count=len(missions))
        return missions


async def get_missions_ending_soon(days: int = 3) -> List[Dict[str, Any]]:
    """
    Fetch missions that are ending within the specified days.

    Args:
        days: Number of days until mission ends

    Returns:
        List of mission dictionaries
    """
    async with get_db() as db:
        cutoff_date = datetime.utcnow() + timedelta(days=days)

        query = text("""
            SELECT
                m.id,
                m."userId" as user_id,
                m."walletId" as wallet_id,
                m."endsAt" as ends_at,
                m.status,
                t.address as user_wallet_address
            FROM agent_missions m
            LEFT JOIN turnkey_signers t ON m."turnkeySignerId" = t.id
            WHERE m.status = 'ACTIVE'
            AND m."endsAt" IS NOT NULL
            AND m."endsAt" <= :cutoff_date
            ORDER BY m."endsAt" ASC
        """)

        result = await db.execute(query, {"cutoff_date": cutoff_date})
        rows = result.fetchall()

        return [
            {
                "id": row.id,
                "user_id": row.user_id,
                "wallet_id": row.wallet_id,
                "ends_at": row.ends_at,
                "status": row.status,
                "user_wallet_address": row.user_wallet_address,
            }
            for row in rows
        ]


async def get_missions_by_status(status: str) -> List[Dict[str, Any]]:
    """
    Fetch missions by status.

    Args:
        status: Mission status (DEPOSITING, APPROVING, ACTIVE, etc.)

    Returns:
        List of mission dictionaries
    """
    async with get_db() as db:
        query = text("""
            SELECT
                m.id,
                m."userId" as user_id,
                m."walletId" as wallet_id,
                m.strategy,
                m."riskLevel" as risk_level,
                m."initialCapital" as initial_capital,
                m."durationDays" as duration_days,
                m."maxLeverage" as max_leverage,
                m."allowedAssets" as allowed_assets,
                m.status,
                m."startedAt" as started_at,
                m."endsAt" as ends_at,
                m."hyperliquidApproved" as hyperliquid_approved,
                m."agentAddress" as agent_address,
                m."masterEoaAddress" as master_eoa_address,
                m."masterEoaKeyEnc" as master_eoa_key_enc,
                m.metadata,
                t.address as user_wallet_address
            FROM agent_missions m
            LEFT JOIN turnkey_signers t ON m."turnkeySignerId" = t.id
            WHERE m.status = :status
            ORDER BY m."createdAt" ASC
        """)

        result = await db.execute(query, {"status": status})
        rows = result.fetchall()

        return [
            {
                "id": row.id,
                "user_id": row.user_id,
                "wallet_id": row.wallet_id,
                "strategy": row.strategy,
                "risk_level": row.risk_level,
                "initial_capital": float(row.initial_capital) if row.initial_capital else 0,
                "duration_days": row.duration_days,
                "max_leverage": row.max_leverage,
                "allowed_assets": row.allowed_assets,
                "status": row.status,
                "started_at": row.started_at,
                "ends_at": row.ends_at,
                "hyperliquid_approved": row.hyperliquid_approved,
                "agent_address": row.agent_address,
                "master_eoa_address": getattr(row, "master_eoa_address", None),
                "master_eoa_key_enc": getattr(row, "master_eoa_key_enc", None),
                "metadata": row.metadata,
                "user_wallet_address": row.user_wallet_address,
            }
            for row in rows
        ]


async def get_active_missions_with_wallets() -> List[Dict[str, Any]]:
    """
    Fetch active missions with wallet addresses for PnL sync.

    Returns:
        List of missions with wallet details
    """
    return await get_active_missions()


async def update_position(
    position_id: str,
    current_price: float,
    unrealized_pnl: float,
    funding_paid: Optional[float] = None,
) -> bool:
    """
    Update a position's current price and PnL.

    Args:
        position_id: Position ID
        current_price: Current market price
        unrealized_pnl: Calculated unrealized PnL
        funding_paid: Cumulative funding paid (optional)

    Returns:
        True if update was successful
    """
    async with get_db() as db:
        update_data = {
            "currentPrice": current_price,
            "unrealizedPnl": unrealized_pnl,
            "updatedAt": datetime.utcnow(),
        }

        if funding_paid is not None:
            update_data["fundingPaid"] = funding_paid

        query = text("""
            UPDATE agent_positions
            SET "currentPrice" = :current_price,
                "unrealizedPnl" = :unrealized_pnl,
                "fundingPaid" = COALESCE(:funding_paid, "fundingPaid")
            WHERE id = :position_id
        """)

        result = await db.execute(query, {
            "position_id": position_id,
            "current_price": current_price,
            "unrealized_pnl": unrealized_pnl,
            "funding_paid": funding_paid,
        })

        return result.rowcount > 0


async def update_mission_pnl(
    mission_id: str,
    current_value: float,
    total_pnl: float,
    max_drawdown: Optional[float] = None,
) -> bool:
    """
    Update a mission's current value and PnL metrics.

    Args:
        mission_id: Mission ID
        current_value: Current total value
        total_pnl: Total PnL
        max_drawdown: Max drawdown percentage (optional)

    Returns:
        True if update was successful
    """
    async with get_db() as db:
        query = text("""
            UPDATE agent_missions
            SET "currentValue" = :current_value,
                "totalPnl" = :total_pnl,
                "maxDrawdown" = COALESCE(:max_drawdown, "maxDrawdown"),
                "updatedAt" = NOW()
            WHERE id = :mission_id
        """)

        result = await db.execute(query, {
            "mission_id": mission_id,
            "current_value": current_value,
            "total_pnl": total_pnl,
            "max_drawdown": max_drawdown,
        })

        return result.rowcount > 0


async def record_trade_execution(
    mission_id: str,
    action: str,
    asset: str,
    quantity: float,
    price: Optional[float],
    mission_day: int,
    decision_reason: str,
    user_balance: float,
    success: bool,
    tx_hash: Optional[str] = None,
    error_message: Optional[str] = None,
    signal_id: Optional[str] = None,
    dry_run: bool = False,
) -> str:
    """
    Record a trade execution in the database.

    Returns:
        The created execution ID
    """
    import uuid

    async with get_db() as db:
        execution_id = str(uuid.uuid4())

        query = text("""
            INSERT INTO agent_trade_executions (
                id, "missionId", "signalId", action, asset, quantity, price,
                "missionDay", "decisionReason", "userBalanceAtTime",
                success, "hyperliquidTxHash", "errorMessage", "isDryRun", "executedAt"
            ) VALUES (
                :id, :mission_id, :signal_id, :action, :asset, :quantity, :price,
                :mission_day, :decision_reason, :user_balance,
                :success, :tx_hash, :error_message, :is_dry_run, NOW()
            )
        """)

        await db.execute(query, {
            "id": execution_id,
            "mission_id": mission_id,
            "signal_id": signal_id,
            "action": action,
            "asset": asset,
            "quantity": quantity,
            "price": price,
            "mission_day": mission_day,
            "decision_reason": decision_reason,
            "user_balance": user_balance,
            "success": success,
            "tx_hash": tx_hash,
            "error_message": error_message,
            "is_dry_run": dry_run,
        })

        logger.info(
            "Trade execution recorded",
            execution_id=execution_id,
            mission_id=mission_id,
            action=action,
            success=success,
        )

        return execution_id


async def create_pnl_snapshot(
    mission_id: str,
    total_value: float,
    total_pnl: float,
    unrealized_pnl: float,
    realized_pnl: float,
) -> str:
    """
    Create a PnL snapshot for a mission.

    Returns:
        The created snapshot ID
    """
    import uuid

    async with get_db() as db:
        snapshot_id = str(uuid.uuid4())

        query = text("""
            INSERT INTO agent_pnl_snapshots (
                id, "missionId", timestamp, "totalValue", "totalPnl",
                "unrealizedPnl", "realizedPnl"
            ) VALUES (
                :id, :mission_id, NOW(), :total_value, :total_pnl,
                :unrealized_pnl, :realized_pnl
            )
        """)

        await db.execute(query, {
            "id": snapshot_id,
            "mission_id": mission_id,
            "total_value": total_value,
            "total_pnl": total_pnl,
            "unrealized_pnl": unrealized_pnl,
            "realized_pnl": realized_pnl,
        })

        return snapshot_id


async def record_agent_audit(
    node: str,
    action: str,
    mission_id: Optional[str] = None,
    user_id: Optional[str] = None,
    asset: Optional[str] = None,
    llm_prompt: Optional[str] = None,
    llm_response: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_tokens: Optional[int] = None,
    decision: Optional[Dict[str, Any]] = None,
    reasoning: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    success: bool = True,
    error_message: Optional[str] = None,
    dry_run: bool = False,
) -> str:
    """
    Record an agent audit log entry for transparency and debugging.

    Args:
        node: Pipeline node (market_analysis, user_filter, execution, monitoring, lifecycle)
        action: Action type (llm_call, signal_generated, signal_skipped, filter_decision, etc.)
        mission_id: Associated mission ID
        user_id: Associated user ID
        asset: Asset being traded
        llm_prompt: The prompt sent to the LLM
        llm_response: The raw LLM response
        llm_model: LLM model used
        llm_tokens: Tokens consumed
        decision: Structured decision data (JSON)
        reasoning: Human-readable reasoning
        metadata: Additional context
        success: Whether the action succeeded
        error_message: Error details if failed
        dry_run: Whether this was a dry run

    Returns:
        The created audit log ID
    """
    import json
    import uuid

    async with get_db() as db:
        audit_id = str(uuid.uuid4())

        # Serialize decision and metadata to JSON strings for JSONB columns
        decision_json = json.dumps(decision) if decision else None
        metadata_json = json.dumps(metadata) if metadata else None

        query = text("""
            INSERT INTO agent_audit_logs (
                id, "missionId", "userId", node, action, asset,
                "llmPrompt", "llmResponse", "llmModel", "llmTokens",
                decision, reasoning, metadata,
                success, "errorMessage", "dryRun", "createdAt"
            ) VALUES (
                :id, :mission_id, :user_id, :node, :action, :asset,
                :llm_prompt, :llm_response, :llm_model, :llm_tokens,
                :decision::jsonb, :reasoning, :metadata::jsonb,
                :success, :error_message, :dry_run, NOW()
            )
        """)

        await db.execute(query, {
            "id": audit_id,
            "mission_id": mission_id,
            "user_id": user_id,
            "node": node,
            "action": action,
            "asset": asset,
            "llm_prompt": llm_prompt,
            "llm_response": llm_response,
            "llm_model": llm_model,
            "llm_tokens": llm_tokens,
            "decision": decision_json,
            "reasoning": reasoning,
            "metadata": metadata_json,
            "success": success,
            "error_message": error_message,
            "dry_run": dry_run,
        })

        logger.debug(
            "Agent audit recorded",
            audit_id=audit_id,
            node=node,
            action=action,
            mission_id=mission_id,
        )

        return audit_id


async def save_signal(signal: Dict[str, Any], cycle_id: str) -> str:
    """
    Persist a generated trading signal to the agent_signals table.

    Args:
        signal: Signal dict from signal_provider
        cycle_id: Trading cycle ID

    Returns:
        The created signal ID
    """
    import uuid

    async with get_db() as db:
        signal_db_id = str(uuid.uuid4())
        signal_id = signal.get("signal_id") or f"{cycle_id}_{signal['asset'][:3]}"

        # Signals expire after 15 minutes (one trading cycle)
        expires_at = datetime.utcnow() + timedelta(minutes=15)

        query = text("""
            INSERT INTO agent_signals (
                id, "signalId", direction, asset, confidence,
                "recommendedLeverage", reasoning, "strategyTag",
                "ragContextIds", "maxDrawdown30d", "volatilityScore",
                "generatedAt", "expiresAt", "isProcessed"
            ) VALUES (
                :id, :signal_id, :direction, :asset, :confidence,
                :leverage, :reasoning, :strategy_tag,
                :rag_ids, :max_dd, :vol_score,
                :generated_at, :expires_at, false
            )
            ON CONFLICT ("signalId") DO UPDATE SET
                direction = EXCLUDED.direction,
                confidence = EXCLUDED.confidence,
                reasoning = EXCLUDED.reasoning
        """)

        generated_at = signal.get("generated_at")
        if isinstance(generated_at, str):
            generated_at = datetime.fromisoformat(generated_at.replace("Z", "+00:00")).replace(tzinfo=None)
        elif generated_at is None:
            generated_at = datetime.utcnow()

        await db.execute(query, {
            "id": signal_db_id,
            "signal_id": signal_id,
            "direction": signal.get("direction", "LONG"),
            "asset": signal.get("asset", ""),
            "confidence": signal.get("confidence", "LOW"),
            "leverage": signal.get("recommended_leverage", 1),
            "reasoning": signal.get("reasoning", ""),
            "strategy_tag": signal.get("strategy_tag", "unknown"),
            "rag_ids": signal.get("rag_context_ids", []),
            "max_dd": signal.get("max_drawdown_30d"),
            "vol_score": signal.get("volatility_score"),
            "generated_at": generated_at,
            "expires_at": expires_at,
        })

        logger.info(
            "Signal persisted",
            signal_id=signal_id,
            asset=signal.get("asset"),
            direction=signal.get("direction"),
        )

        return signal_id


async def get_signals(
    asset: Optional[str] = None,
    direction: Optional[str] = None,
    confidence: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """Fetch persisted signals with optional filters."""
    async with get_db() as db:
        conditions = []
        params: Dict[str, Any] = {"limit": limit, "offset": offset}

        if asset:
            conditions.append('asset = :asset')
            params["asset"] = asset
        if direction:
            conditions.append('direction = :direction')
            params["direction"] = direction
        if confidence:
            conditions.append('confidence = :confidence')
            params["confidence"] = confidence

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = text(f"""
            SELECT id, "signalId", direction, asset, confidence,
                   "recommendedLeverage", reasoning, "strategyTag",
                   "ragContextIds", "maxDrawdown30d", "volatilityScore",
                   "generatedAt", "expiresAt", "isProcessed", "processedAt",
                   "usersNotified", "ordersGenerated", "ordersExecuted"
            FROM agent_signals
            WHERE {where_clause}
            ORDER BY "generatedAt" DESC
            LIMIT :limit OFFSET :offset
        """)

        result = await db.execute(query, params)
        rows = result.fetchall()

        return [
            {
                "id": row.id,
                "signal_id": row.signalId,
                "asset": row.asset,
                "direction": row.direction,
                "confidence": row.confidence,
                "recommended_leverage": row.recommendedLeverage,
                "reasoning": row.reasoning,
                "strategy_tag": row.strategyTag,
                "rag_context_ids": row.ragContextIds,
                "max_drawdown_30d": row.maxDrawdown30d,
                "volatility_score": row.volatilityScore,
                "generated_at": row.generatedAt,
                "expires_at": row.expiresAt,
                "is_processed": row.isProcessed,
                "processed_at": row.processedAt,
                "users_notified": row.usersNotified,
                "orders_generated": row.ordersGenerated,
                "orders_executed": row.ordersExecuted,
            }
            for row in rows
        ]


async def get_active_signals() -> List[Dict[str, Any]]:
    """Fetch non-expired signals."""
    async with get_db() as db:
        query = text("""
            SELECT id, "signalId", direction, asset, confidence,
                   "recommendedLeverage", reasoning, "strategyTag",
                   "ragContextIds", "maxDrawdown30d", "volatilityScore",
                   "generatedAt", "expiresAt", "isProcessed", "processedAt",
                   "usersNotified", "ordersGenerated", "ordersExecuted"
            FROM agent_signals
            WHERE "expiresAt" > NOW()
            ORDER BY "generatedAt" DESC
        """)

        result = await db.execute(query)
        rows = result.fetchall()

        return [
            {
                "id": row.id,
                "signal_id": row.signalId,
                "asset": row.asset,
                "direction": row.direction,
                "confidence": row.confidence,
                "recommended_leverage": row.recommendedLeverage,
                "reasoning": row.reasoning,
                "strategy_tag": row.strategyTag,
                "rag_context_ids": row.ragContextIds,
                "max_drawdown_30d": row.maxDrawdown30d,
                "volatility_score": row.volatilityScore,
                "generated_at": row.generatedAt,
                "expires_at": row.expiresAt,
                "is_processed": row.isProcessed,
                "processed_at": row.processedAt,
                "users_notified": row.usersNotified,
                "orders_generated": row.ordersGenerated,
                "orders_executed": row.ordersExecuted,
            }
            for row in rows
        ]


async def get_signal_stats() -> Dict[str, Any]:
    """Get signal statistics for the API."""
    async with get_db() as db:
        query = text("""
            SELECT
                COUNT(*) FILTER (WHERE "generatedAt" >= CURRENT_DATE) AS total_signals_today,
                COUNT(*) FILTER (WHERE direction = 'LONG' AND "generatedAt" >= CURRENT_DATE) AS long_signals,
                COUNT(*) FILTER (WHERE direction = 'SHORT' AND "generatedAt" >= CURRENT_DATE) AS short_signals,
                COUNT(*) FILTER (WHERE confidence = 'HIGH' AND "generatedAt" >= CURRENT_DATE) AS high_confidence,
                COALESCE(
                    AVG("ordersExecuted"::float / NULLIF("ordersGenerated", 0))
                    FILTER (WHERE "generatedAt" >= NOW() - INTERVAL '7 days'),
                    0
                ) AS success_rate_7d
            FROM agent_signals
        """)

        result = await db.execute(query)
        row = result.fetchone()

        return {
            "total_signals_today": row.total_signals_today or 0,
            "long_signals": row.long_signals or 0,
            "short_signals": row.short_signals or 0,
            "high_confidence": row.high_confidence or 0,
            "success_rate_7d": float(row.success_rate_7d or 0),
            "avg_return_per_signal": 0,  # requires trade PnL correlation
        }


async def update_signal_metrics(
    signal_id: str,
    orders_generated: int = 0,
    orders_executed: int = 0,
) -> None:
    """Update processing metrics on a signal after execution."""
    async with get_db() as db:
        query = text("""
            UPDATE agent_signals
            SET "isProcessed" = true,
                "processedAt" = NOW(),
                "ordersGenerated" = "ordersGenerated" + :orders_generated,
                "ordersExecuted" = "ordersExecuted" + :orders_executed
            WHERE "signalId" = :signal_id
        """)

        await db.execute(query, {
            "signal_id": signal_id,
            "orders_generated": orders_generated,
            "orders_executed": orders_executed,
        })


async def create_position(
    mission_id: str,
    asset: str,
    direction: str,
    entry_price: float,
    quantity: float,
    leverage: int,
    margin_used: float,
    signal_id: Optional[str] = None,
    hyperliquid_order_id: Optional[str] = None,
    liquidation_price: Optional[float] = None,
) -> str:
    """
    Create a new open position in the database.

    Called after a successful ENTER trade on Hyperliquid.

    Returns:
        The created position ID
    """
    import uuid

    async with get_db() as db:
        position_id = str(uuid.uuid4())

        query = text("""
            INSERT INTO agent_positions (
                id, "missionId", "signalId", asset, direction,
                "entryPrice", "currentPrice", quantity, leverage,
                "hyperliquidOrderId", "marginUsed", "liquidationPrice",
                "unrealizedPnl", "realizedPnl", "fundingPaid",
                status, "openedAt"
            ) VALUES (
                :id, :mission_id, :signal_id, :asset, :direction,
                :entry_price, :entry_price, :quantity, :leverage,
                :hl_order_id, :margin_used, :liquidation_price,
                0, 0, 0,
                'OPEN', NOW()
            )
        """)

        await db.execute(query, {
            "id": position_id,
            "mission_id": mission_id,
            "signal_id": signal_id,
            "asset": asset,
            "direction": direction,
            "entry_price": entry_price,
            "quantity": quantity,
            "leverage": leverage,
            "hl_order_id": hyperliquid_order_id,
            "margin_used": margin_used,
            "liquidation_price": liquidation_price,
        })

        logger.info(
            "Position created",
            position_id=position_id,
            mission_id=mission_id,
            asset=asset,
            direction=direction,
        )

        return position_id


async def set_position_risk_levels(
    position_id: str,
    stop_loss_price: float,
    take_profit_price: float,
) -> None:
    """
    Set the stop loss and take profit prices for a position.
    Called after create_position() to persist risk levels from the user's risk profile.
    """
    async with get_db() as db:
        query = text("""
            UPDATE agent_positions
            SET "stopLossPrice" = :sl_price,
                "takeProfitPrice" = :tp_price
            WHERE id = :position_id
        """)

        await db.execute(query, {
            "position_id": position_id,
            "sl_price": stop_loss_price,
            "tp_price": take_profit_price,
        })

        logger.info(
            "Position risk levels set",
            position_id=position_id,
            stop_loss=stop_loss_price,
            take_profit=take_profit_price,
        )


async def close_position(
    position_id: str,
    close_price: float,
    realized_pnl: float,
    close_reason: str = "SIGNAL",
) -> bool:
    """
    Close an open position.

    Called after a successful EXIT trade on Hyperliquid.

    Args:
        position_id: Position ID to close
        close_price: Price at which the position was closed
        realized_pnl: Realized PnL from this position
        close_reason: Why the position was closed (SIGNAL, STOP_LOSS, TAKE_PROFIT, FORCE_CLOSE, LIQUIDATION)

    Returns:
        True if the position was closed successfully
    """
    async with get_db() as db:
        query = text("""
            UPDATE agent_positions
            SET status = 'CLOSED',
                "currentPrice" = :close_price,
                "realizedPnl" = :realized_pnl,
                "unrealizedPnl" = 0,
                "closedAt" = NOW(),
                "closeReason" = :close_reason
            WHERE id = :position_id
            AND status = 'OPEN'
        """)

        result = await db.execute(query, {
            "position_id": position_id,
            "close_price": close_price,
            "realized_pnl": realized_pnl,
            "close_reason": close_reason,
        })

        if result.rowcount > 0:
            logger.info(
                "Position closed",
                position_id=position_id,
                realized_pnl=realized_pnl,
                close_reason=close_reason,
            )
            return True

        logger.warning("Position not found or already closed", position_id=position_id)
        return False


async def get_open_positions(mission_id: str) -> List[Dict[str, Any]]:
    """
    Get all open positions for a mission.

    Args:
        mission_id: Mission ID

    Returns:
        List of position dictionaries
    """
    async with get_db() as db:
        query = text("""
            SELECT
                id, "missionId", "signalId", asset, direction,
                "entryPrice", "currentPrice", quantity, leverage,
                "hyperliquidOrderId", "marginUsed", "liquidationPrice",
                "unrealizedPnl", "realizedPnl", "fundingPaid",
                "stopLossPrice", "takeProfitPrice",
                status, "openedAt"
            FROM agent_positions
            WHERE "missionId" = :mission_id
            AND status = 'OPEN'
        """)

        result = await db.execute(query, {"mission_id": mission_id})
        rows = result.fetchall()

        return [
            {
                "id": row.id,
                "mission_id": row.missionId,
                "signal_id": row.signalId,
                "asset": row.asset,
                "direction": row.direction,
                "entry_price": float(row.entryPrice),
                "current_price": float(row.currentPrice),
                "quantity": float(row.quantity),
                "leverage": row.leverage,
                "hyperliquid_order_id": row.hyperliquidOrderId,
                "margin_used": float(row.marginUsed),
                "liquidation_price": float(row.liquidationPrice) if row.liquidationPrice else None,
                "unrealized_pnl": float(row.unrealizedPnl),
                "realized_pnl": float(row.realizedPnl),
                "funding_paid": float(row.fundingPaid),
                "stop_loss_price": float(row.stopLossPrice) if row.stopLossPrice else None,
                "take_profit_price": float(row.takeProfitPrice) if row.takeProfitPrice else None,
                "status": row.status,
                "opened_at": row.openedAt,
            }
            for row in rows
        ]


# ==================
# User-Facing Query Functions (Phase 4)
# ==================

async def get_mission_by_id(mission_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Fetch a single mission by ID, optionally scoped to a user."""
    async with get_db() as db:
        params = {"mission_id": mission_id}
        user_clause = ""
        if user_id:
            user_clause = 'AND m."userId" = :user_id'
            params["user_id"] = user_id

        query = text(f"""
            SELECT
                m.id,
                m."userId" as user_id,
                m."walletId" as wallet_id,
                m."hyperliquidApproved" as hyperliquid_approved,
                m.strategy,
                m."riskLevel" as risk_level,
                m."durationDays" as duration_days,
                m."startedAt" as started_at,
                m."endsAt" as ends_at,
                m."initialCapital" as initial_capital,
                m."currentValue" as current_value,
                m."maxLeverage" as max_leverage,
                m."allowedAssets" as allowed_assets,
                m."agentAddress" as agent_address,
                m.status,
                m."pausedAt" as paused_at,
                m."completedAt" as completed_at,
                m."totalPnl" as total_pnl,
                m."totalTrades" as total_trades,
                m."winRate" as win_rate,
                m."maxDrawdown" as max_drawdown,
                m.metadata,
                m."createdAt" as created_at,
                m."masterEoaAddress" as master_eoa_address,
                m."masterEoaKeyEnc" as master_eoa_key_enc,
                m."vaultKeyCreated" as vault_key_created,
                m."vaultKeyDestroyed" as vault_key_destroyed,
                t.address as user_wallet_address,
                (SELECT ka.address FROM kernel_accounts ka
                 WHERE ka."walletId" = m."walletId" LIMIT 1) as kernel_account_address
            FROM agent_missions m
            LEFT JOIN turnkey_signers t ON m."turnkeySignerId" = t.id
            WHERE m.id = :mission_id
            {user_clause}
        """)

        result = await db.execute(query, params)
        row = result.fetchone()

        if not row:
            return None

        initial_capital = float(row.initial_capital) if row.initial_capital else 0
        current_value = float(row.current_value) if row.current_value else None
        total_pnl = float(row.total_pnl) if row.total_pnl else 0
        total_pnl_percent = (
            ((current_value - initial_capital) / initial_capital * 100)
            if initial_capital > 0 and current_value is not None
            else 0
        )

        days_remaining = None
        day_number = 1
        if row.ends_at:
            now = datetime.utcnow()
            ends = row.ends_at.replace(tzinfo=None) if row.ends_at.tzinfo else row.ends_at
            days_remaining = max(0, (ends - now).days)
        if row.started_at:
            started = row.started_at.replace(tzinfo=None) if row.started_at.tzinfo else row.started_at
            day_number = max(1, (datetime.utcnow() - started).days + 1)

        return {
            "id": row.id,
            "user_id": row.user_id,
            "wallet_id": row.wallet_id,
            "hyperliquid_approved": row.hyperliquid_approved,
            "strategy": row.strategy,
            "risk_level": row.risk_level,
            "duration_days": row.duration_days,
            "started_at": row.started_at,
            "ends_at": row.ends_at,
            "initial_capital": initial_capital,
            "current_value": current_value,
            "max_leverage": row.max_leverage,
            "allowed_assets": row.allowed_assets or ["ETH-USD", "BTC-USD"],
            "status": row.status,
            "total_pnl": total_pnl,
            "total_pnl_percent": round(total_pnl_percent, 2),
            "total_trades": row.total_trades or 0,
            "win_rate": row.win_rate or 0,
            "max_drawdown": row.max_drawdown or 0,
            "day_number": day_number,
            "days_remaining": days_remaining,
            "agent_address": row.agent_address,
            "user_wallet_address": row.user_wallet_address,
            "kernel_account_address": row.kernel_account_address,
            "master_eoa_address": getattr(row, "master_eoa_address", None),
            "master_eoa_key_enc": getattr(row, "master_eoa_key_enc", None),
            "vault_key_created": getattr(row, "vault_key_created", False),
            "vault_key_destroyed": getattr(row, "vault_key_destroyed", False),
            "created_at": row.created_at,
        }


async def get_missions_by_user(
    user_id: str,
    status_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fetch all missions for a user, optionally filtered by status."""
    async with get_db() as db:
        params: Dict[str, Any] = {"user_id": user_id}
        status_clause = ""
        if status_filter:
            status_clause = "AND m.status = :status"
            params["status"] = status_filter

        query = text(f"""
            SELECT
                m.id, m.status, m.strategy,
                m."durationDays" as duration_days,
                m."startedAt" as started_at,
                m."initialCapital" as initial_capital,
                m."currentValue" as current_value,
                m."totalPnl" as total_pnl
            FROM agent_missions m
            WHERE m."userId" = :user_id
            {status_clause}
            ORDER BY m."createdAt" DESC
        """)

        result = await db.execute(query, params)
        rows = result.fetchall()

        missions = []
        for row in rows:
            ic = float(row.initial_capital) if row.initial_capital else 0
            cv = float(row.current_value) if row.current_value else ic
            tp = float(row.total_pnl) if row.total_pnl else 0
            pnl_pct = ((cv - ic) / ic * 100) if ic > 0 else 0

            day_number = 1
            days_remaining = row.duration_days or 30
            if row.started_at:
                started = row.started_at.replace(tzinfo=None) if row.started_at.tzinfo else row.started_at
                day_number = max(1, (datetime.utcnow() - started).days + 1)
                days_remaining = max(0, (row.duration_days or 30) - day_number + 1)

            missions.append({
                "id": row.id,
                "status": row.status,
                "strategy": row.strategy,
                "day_number": day_number,
                "days_remaining": days_remaining,
                "initial_capital": str(ic),
                "current_value": str(cv),
                "total_pnl": str(tp),
                "total_pnl_percent": round(pnl_pct, 2),
            })

        return missions


_ALLOWED_EXTRA_FIELDS = frozenset({
    "vaultKeyDestroyed", "vaultKeyCreated", "finalBalance", "protocolFee",
    "userPayout", "pausedAt", "completedAt", "startedAt", "endsAt",
    "hyperliquidApproved", "feeCollectedAt", "payoutCompletedAt",
    "masterEoaKeyEnc", "agentPrivateKeyEnc", "agentAddress",
})


async def update_mission_status(
    mission_id: str,
    new_status: str,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> bool:
    """Update a mission's status and optional extra fields.

    Column names in extra_fields are validated against an allowlist
    to prevent SQL injection via dynamic column interpolation.
    """
    async with get_db() as db:
        set_parts = ['"status" = :new_status', '"updatedAt" = NOW()']
        params: Dict[str, Any] = {"mission_id": mission_id, "new_status": new_status}

        if extra_fields:
            for key, value in extra_fields.items():
                if key not in _ALLOWED_EXTRA_FIELDS:
                    raise ValueError(
                        f"Column '{key}' is not in the allowed fields list "
                        f"for update_mission_status"
                    )
                param_name = f"extra_{key}"
                set_parts.append(f'"{key}" = :{param_name}')
                params[param_name] = value

        set_clause = ", ".join(set_parts)
        query = text(f"""
            UPDATE agent_missions
            SET {set_clause}
            WHERE id = :mission_id
        """)

        result = await db.execute(query, params)
        return result.rowcount > 0


async def claim_mission_for_activation(mission_id: str) -> bool:
    """Atomically transition DEPOSITING → APPROVING.

    Uses a WHERE condition on current status so only one concurrent
    caller can succeed — prevents race conditions between deposit
    polling and manual activation.

    Returns True if this caller won the claim, False if already claimed.
    """
    async with get_db() as db:
        query = text("""
            UPDATE agent_missions
            SET status = 'APPROVING', "updatedAt" = NOW()
            WHERE id = :mission_id AND status = 'DEPOSITING'
        """)
        result = await db.execute(query, {"mission_id": mission_id})
        return result.rowcount > 0


async def count_active_missions_for_user(user_id: str) -> int:
    """Count missions in active/pending states for per-user rate limiting."""
    async with get_db() as db:
        query = text("""
            SELECT COUNT(*) FROM agent_missions
            WHERE "userId" = :user_id
              AND status IN ('PENDING', 'DEPOSITING', 'APPROVING', 'ACTIVE')
        """)
        result = await db.execute(query, {"user_id": user_id})
        row = result.scalar()
        return int(row) if row else 0


async def get_agent_key_ciphertext(mission_id: str) -> Optional[str]:
    """Get the Vault-encrypted agent key for local signing.

    Returns the Vault ciphertext (vault:v1:...) or None if not found.
    """
    async with get_db() as db:
        query = text("""
            SELECT "agentPrivateKeyEnc", "agentAddress"
            FROM agent_missions
            WHERE id = :mission_id
        """)
        result = await db.execute(query, {"mission_id": mission_id})
        row = result.fetchone()
        if row and row.agentPrivateKeyEnc:
            return row.agentPrivateKeyEnc
        return None


async def update_mission_vault_key(
    mission_id: str,
    master_eoa_address: str,
    master_eoa_key_enc: str,
) -> bool:
    """Store the Vault-encrypted Master EOA key on a mission."""
    async with get_db() as db:
        query = text("""
            UPDATE agent_missions
            SET "masterEoaAddress" = :master_eoa_address,
                "masterEoaKeyEnc" = :master_eoa_key_enc,
                "vaultKeyCreated" = true,
                "updatedAt" = NOW()
            WHERE id = :mission_id
        """)
        result = await db.execute(query, {
            "mission_id": mission_id,
            "master_eoa_address": master_eoa_address,
            "master_eoa_key_enc": master_eoa_key_enc,
        })
        return result.rowcount > 0


async def get_trade_executions(
    mission_id: str,
    user_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple:
    """Fetch trade executions for a mission with pagination. Returns (trades, total_count)."""
    async with get_db() as db:
        params: Dict[str, Any] = {"mission_id": mission_id, "limit": limit, "offset": offset}
        owner_join = ""
        if user_id:
            owner_join = 'AND m."userId" = :user_id'
            params["user_id"] = user_id

        count_query = text(f"""
            SELECT COUNT(*) as cnt
            FROM agent_trade_executions e
            JOIN agent_missions m ON e."missionId" = m.id
            WHERE e."missionId" = :mission_id {owner_join}
        """)
        total_count = (await db.execute(count_query, params)).scalar() or 0

        query = text(f"""
            SELECT
                e.id, e."signalId" as signal_id, e.action, e.asset,
                e.quantity, e.price, e."missionDay" as mission_day,
                e."decisionReason" as decision_reason,
                e.success, e."hyperliquidTxHash" as tx_hash,
                e."executedAt" as executed_at
            FROM agent_trade_executions e
            JOIN agent_missions m ON e."missionId" = m.id
            WHERE e."missionId" = :mission_id {owner_join}
            ORDER BY e."executedAt" DESC
            LIMIT :limit OFFSET :offset
        """)

        result = await db.execute(query, params)
        rows = result.fetchall()

        trades = [
            {
                "id": row.id,
                "signal_id": row.signal_id,
                "action": row.action,
                "asset": row.asset,
                "quantity": float(row.quantity) if row.quantity else 0,
                "price": float(row.price) if row.price else 0,
                "mission_day": row.mission_day,
                "decision_reason": row.decision_reason,
                "success": row.success,
                "tx_hash": row.tx_hash,
                "executed_at": row.executed_at,
            }
            for row in rows
        ]

        return trades, total_count


async def get_pnl_snapshots(mission_id: str, days: int = 30) -> List[Dict[str, Any]]:
    """Fetch PnL snapshots for a mission."""
    async with get_db() as db:
        cutoff = datetime.utcnow() - timedelta(days=days)
        query = text("""
            SELECT id, timestamp,
                "totalValue" as total_value,
                "totalPnl" as total_pnl,
                "unrealizedPnl" as unrealized_pnl,
                "realizedPnl" as realized_pnl
            FROM agent_pnl_snapshots
            WHERE "missionId" = :mission_id AND timestamp >= :cutoff
            ORDER BY timestamp ASC
        """)
        result = await db.execute(query, {"mission_id": mission_id, "cutoff": cutoff})
        return [
            {
                "timestamp": row.timestamp,
                "total_value": float(row.total_value),
                "total_pnl": float(row.total_pnl),
                "unrealized_pnl": float(row.unrealized_pnl),
                "realized_pnl": float(row.realized_pnl),
            }
            for row in result.fetchall()
        ]


async def get_positions_by_user(
    user_id: str,
    mission_id: Optional[str] = None,
    status_filter: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """Fetch positions for a user with optional filters."""
    async with get_db() as db:
        params: Dict[str, Any] = {"user_id": user_id, "limit": limit, "offset": offset}
        filters = ['m."userId" = :user_id']
        if mission_id:
            filters.append('p."missionId" = :mission_id')
            params["mission_id"] = mission_id
        if status_filter:
            filters.append("p.status = :status")
            params["status"] = status_filter

        where_clause = " AND ".join(filters)
        query = text(f"""
            SELECT
                p.id, p."missionId" as mission_id, p.asset, p.direction,
                p."entryPrice" as entry_price, p."currentPrice" as current_price,
                p.quantity, p.leverage, p."marginUsed" as margin_used,
                p."liquidationPrice" as liquidation_price,
                p."unrealizedPnl" as unrealized_pnl,
                p."realizedPnl" as realized_pnl,
                p."fundingPaid" as funding_paid,
                p.status, p."openedAt" as opened_at,
                p."closedAt" as closed_at, p."closeReason" as close_reason
            FROM agent_positions p
            JOIN agent_missions m ON p."missionId" = m.id
            WHERE {where_clause}
            ORDER BY p."openedAt" DESC
            LIMIT :limit OFFSET :offset
        """)

        result = await db.execute(query, params)
        return [
            {
                "id": row.id,
                "mission_id": row.mission_id,
                "asset": row.asset,
                "direction": row.direction,
                "entry_price": float(row.entry_price),
                "current_price": float(row.current_price),
                "quantity": float(row.quantity),
                "leverage": row.leverage,
                "margin_used": float(row.margin_used),
                "liquidation_price": float(row.liquidation_price) if row.liquidation_price else None,
                "unrealized_pnl": float(row.unrealized_pnl),
                "realized_pnl": float(row.realized_pnl),
                "funding_paid": float(row.funding_paid),
                "status": row.status,
                "opened_at": row.opened_at,
                "closed_at": row.closed_at,
                "close_reason": row.close_reason,
            }
            for row in result.fetchall()
        ]


async def get_user_missions_aggregate(user_id: str) -> Dict[str, Any]:
    """Get aggregated stats across all user missions."""
    async with get_db() as db:
        query = text("""
            SELECT
                COUNT(*) as total_missions,
                COUNT(*) FILTER (WHERE status = 'ACTIVE') as active_missions,
                COALESCE(SUM("initialCapital"), 0) as total_invested,
                COALESCE(SUM("currentValue"), 0) as total_current_value,
                COALESCE(SUM("totalPnl"), 0) as total_pnl,
                COALESCE(SUM("totalTrades"), 0) as total_trades
            FROM agent_missions
            WHERE "userId" = :user_id
        """)

        row = (await db.execute(query, {"user_id": user_id})).fetchone()
        ti = float(row.total_invested) if row.total_invested else 0
        tc = float(row.total_current_value) if row.total_current_value else 0
        tp = float(row.total_pnl) if row.total_pnl else 0
        pct = ((tc - ti) / ti * 100) if ti > 0 else 0

        return {
            "total_missions": row.total_missions or 0,
            "active_missions": row.active_missions or 0,
            "total_invested": ti,
            "total_current_value": tc,
            "total_pnl": tp,
            "total_pnl_percent": round(pct, 2),
            "total_trades": row.total_trades or 0,
        }

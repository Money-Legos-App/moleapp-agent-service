"""
Missions API Endpoints
Handles mission creation, management, and retrieval
"""

from datetime import datetime
from decimal import Decimal
from typing import List, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from app.api.deps import UserInfo, get_current_user, require_verified_user

logger = structlog.get_logger(__name__)
router = APIRouter()


# ==================
# Camel case serialization
# ==================

def _to_camel(string: str) -> str:
    parts = string.split("_")
    return parts[0] + "".join(w.capitalize() for w in parts[1:])


# ==================
# Request/Response Models
# ==================

class CreateMissionRequest(BaseModel):
    """Request to create a new agent mission."""

    wallet_id: str = Field(..., description="User's wallet ID")
    strategy: str = Field(
        default="SHORT_TERM_30D",
        description="Strategy type: SHORT_TERM_30D, LONG_TERM_45D, CUSTOM",
    )
    risk_level: str = Field(
        default="MODERATE",
        description="Risk level: CONSERVATIVE, MODERATE, AGGRESSIVE",
    )
    initial_capital: str = Field(..., description="Initial deposit amount in USDC")
    allowed_assets: List[str] = Field(
        default=["ETH-USD", "BTC-USD"],
        description="List of allowed trading assets",
    )


class MissionResponse(BaseModel):
    """Mission details response."""
    model_config = ConfigDict(alias_generator=_to_camel, populate_by_name=True)

    id: str
    user_id: str
    wallet_id: str
    strategy: str
    risk_level: str
    status: str
    initial_capital: str
    current_value: Optional[str]
    total_pnl: str
    total_pnl_percent: float
    total_trades: int
    win_rate: float
    max_drawdown: float
    allowed_assets: List[str]
    max_leverage: int
    duration_days: int
    started_at: Optional[datetime]
    ends_at: Optional[datetime]
    days_remaining: Optional[int]
    hyperliquid_approved: bool
    created_at: datetime


class MissionSummaryResponse(BaseModel):
    """Simplified mission summary for dashboard."""
    model_config = ConfigDict(alias_generator=_to_camel, populate_by_name=True)

    id: str
    status: str
    strategy: str
    day_number: int
    days_remaining: int
    initial_capital: str
    current_value: str
    total_pnl: str
    total_pnl_percent: float


class PauseMissionRequest(BaseModel):
    """Request to pause a mission."""

    reason: Optional[str] = None


# ==================
# Endpoints
# ==================

@router.post("", response_model=MissionResponse, status_code=status.HTTP_201_CREATED)
async def create_mission(
    request: CreateMissionRequest,
    user: UserInfo = Depends(require_verified_user),
):
    """
    Create a new agent mission.

    This initiates the mission creation process:
    1. Validates user's wallet
    2. Creates mission record in database
    3. Returns mission details (user still needs to approve agent on Hyperliquid)
    """
    from app.services.wallet import TurnkeyBridge

    logger.info(
        "Creating mission",
        user_id=user.user_id,
        wallet_id=request.wallet_id,
        strategy=request.strategy,
        initial_capital=request.initial_capital,
    )

    # Validate initial capital
    try:
        capital = Decimal(request.initial_capital)
        if capital < 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Minimum deposit is 10 USDC",
            )
        if capital > 10000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum deposit is 10,000 USDC",
            )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid initial capital amount",
        )

    # Per-user concurrent mission limit (prevents gas station drain / paymaster abuse)
    MAX_ACTIVE_MISSIONS = 5
    from app.services.database import count_active_missions_for_user
    active_count = await count_active_missions_for_user(user.user_id)
    if active_count >= MAX_ACTIVE_MISSIONS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Maximum {MAX_ACTIVE_MISSIONS} concurrent missions allowed",
        )

    # Validate strategy
    valid_strategies = ["SHORT_TERM_30D", "LONG_TERM_45D", "CUSTOM"]
    if request.strategy not in valid_strategies:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid strategy. Must be one of: {valid_strategies}",
        )

    # Validate risk level
    valid_risk_levels = ["CONSERVATIVE", "MODERATE", "AGGRESSIVE"]
    if request.risk_level not in valid_risk_levels:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid risk level. Must be one of: {valid_risk_levels}",
        )

    # Create mission via wallet-service (includes wallet ownership validation)
    try:
        bridge = TurnkeyBridge()
        result = await bridge.create_mission(
            user_id=user.user_id,
            mission_type=request.strategy,
            deposit_amount=request.initial_capital,
            wallet_id=request.wallet_id,
            risk_level=request.risk_level,
        )
        await bridge.close()

        if not result.get("success", False):
            error_code = result.get("error", "")
            # Propagate wallet ownership / authorization errors
            if error_code == "FORBIDDEN":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Wallet does not belong to this user",
                )
            if error_code == "NOT_FOUND":
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=result.get("message", "Wallet or signer not found"),
                )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("message", "Failed to create mission"),
            )

        mission = result.get("mission", {})
        mission_id_created = mission.get("id", "")

        # Generate Master EOA locally + encrypt via Vault Transit
        master_eoa_address = None
        try:
            from app.config import get_settings
            settings = get_settings()

            if settings.vault_configured:
                from eth_account import Account
                from app.services.vault import VaultEncryptionService

                vault = VaultEncryptionService(
                    vault_url=settings.hcp_vault_url,
                    vault_token=settings.hcp_vault_token,
                    namespace=settings.hcp_vault_namespace,
                    mount_point=settings.vault_transit_mount,
                    role_id=settings.vault_role_id,
                    secret_id=settings.vault_secret_id,
                )

                # Ensure the shared encryption key exists
                await vault.ensure_encryption_key()

                # Generate Master EOA locally
                account = Account.create()
                master_eoa_address = account.address

                # Encrypt private key via Vault Transit (aes256-gcm96)
                encrypted_key = await vault.encrypt_private_key(account.key.hex())

                # Clear raw key from memory immediately
                del account

                # Store encrypted key + address in DB
                from app.services.database import update_mission_vault_key
                await update_mission_vault_key(
                    mission_id=mission_id_created,
                    master_eoa_address=master_eoa_address,
                    master_eoa_key_enc=encrypted_key,
                )

                logger.info(
                    "Vault-encrypted Master EOA created for mission",
                    mission_id=mission_id_created,
                    master_eoa=master_eoa_address,
                )
            else:
                logger.info("Vault not configured — skipping Master EOA creation")
        except Exception as vault_err:
            logger.error(
                "Vault key creation failed (mission still created)",
                mission_id=mission_id_created,
                error=str(vault_err),
            )

        # Determine duration
        duration_days = 30 if request.strategy == "SHORT_TERM_30D" else 45

        return MissionResponse(
            id=mission.get("id"),
            user_id=user.user_id,
            wallet_id=request.wallet_id,
            strategy=request.strategy,
            risk_level=request.risk_level,
            status="PENDING",
            initial_capital=request.initial_capital,
            current_value=None,
            total_pnl="0",
            total_pnl_percent=0,
            total_trades=0,
            win_rate=0,
            max_drawdown=0,
            allowed_assets=request.allowed_assets,
            max_leverage=_get_max_leverage(request.risk_level),
            duration_days=duration_days,
            started_at=None,
            ends_at=None,
            days_remaining=duration_days,
            hyperliquid_approved=False,
            created_at=datetime.utcnow(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Mission creation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create mission",
        )


@router.get("", response_model=List[MissionSummaryResponse])
async def list_missions(
    user: UserInfo = Depends(get_current_user),
    status_filter: Optional[str] = None,
):
    """
    List all missions for the current user.

    Optional filter by status: PENDING, ACTIVE, PAUSED, COMPLETED, LIQUIDATED, REVOKED
    """
    from app.services.database import get_missions_by_user

    logger.info("Listing missions", user_id=user.user_id, status=status_filter)

    missions = await get_missions_by_user(user.user_id, status_filter)
    return [MissionSummaryResponse(**m) for m in missions]


@router.get("/{mission_id}", response_model=MissionResponse)
async def get_mission(
    mission_id: str,
    user: UserInfo = Depends(get_current_user),
):
    """Get detailed mission information."""
    from app.services.database import get_mission_by_id

    logger.info("Getting mission", mission_id=mission_id, user_id=user.user_id)

    mission = await get_mission_by_id(mission_id, user.user_id)
    if not mission:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Mission not found",
        )

    return MissionResponse(
        id=mission["id"],
        user_id=mission["user_id"],
        wallet_id=mission["wallet_id"],
        strategy=mission["strategy"],
        risk_level=mission["risk_level"],
        status=mission["status"],
        initial_capital=str(mission["initial_capital"]),
        current_value=str(mission["current_value"]) if mission["current_value"] else None,
        total_pnl=str(mission["total_pnl"]),
        total_pnl_percent=mission["total_pnl_percent"],
        total_trades=mission["total_trades"],
        win_rate=mission["win_rate"],
        max_drawdown=mission["max_drawdown"],
        allowed_assets=mission["allowed_assets"],
        max_leverage=mission["max_leverage"],
        duration_days=mission["duration_days"],
        started_at=mission["started_at"],
        ends_at=mission["ends_at"],
        days_remaining=mission["days_remaining"],
        hyperliquid_approved=mission["hyperliquid_approved"],
        created_at=mission["created_at"],
    )


@router.post("/{mission_id}/activate")
async def activate_mission(
    mission_id: str,
    user: UserInfo = Depends(require_verified_user),
):
    """
    Activate a pending mission.

    Triggers the deposit + agent approval flow:
    1. Validate mission is PENDING and owned by user
    2. Initiate USDC deposit to Hyperliquid via AA bridge
    3. Mission transitions: PENDING → DEPOSITING → APPROVING → ACTIVE
    """
    from app.services.database import get_mission_by_id, update_mission_status
    from app.services.wallet import TurnkeyBridge

    logger.info("Activating mission", mission_id=mission_id, user_id=user.user_id)

    # Fetch and validate
    mission = await get_mission_by_id(mission_id, user.user_id)
    if not mission:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Mission not found",
        )

    if mission["status"] != "PENDING":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Mission cannot be activated from '{mission['status']}' status. Must be PENDING.",
        )

    from app.config import get_settings
    settings = get_settings()

    try:
        master_eoa = mission.get("master_eoa_address", "")

        if settings.is_testnet:
            # =============================================
            # TESTNET PATH: Auto-fund Master EOA via treasury,
            # then approve agent via Vault and activate.
            # =============================================
            from app.services.hyperliquid import HyperliquidClient

            hl_client = HyperliquidClient()
            bridge = TurnkeyBridge()

            # Destination: Master EOA (Vault-managed) or fallback to user EOA
            destination = master_eoa or mission.get("user_wallet_address", "")
            amount = float(mission["initial_capital"])

            try:
                # Step 1: Auto-fund Master EOA on HL L1 via treasury usdSend
                if settings.hl_treasury_private_key:
                    logger.info(
                        "Testnet: auto-funding Master EOA via treasury",
                        mission_id=mission_id,
                        destination=destination,
                        amount=amount,
                    )
                    fund_result = await hl_client.treasury_send_usd(
                        destination=destination,
                        amount=amount,
                    )
                    if not fund_result.get("success"):
                        raise HTTPException(
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Treasury auto-fund failed: {fund_result.get('error')}",
                        )

                # Step 2: Approve agent via SDK (or legacy Turnkey)
                from app.tasks.deposit_polling import _approve_and_activate_via_vault

                activated = await _approve_and_activate_via_vault(
                    mission_id=mission_id,
                    encrypted_master_key=mission.get("master_eoa_key_enc", ""),
                    wallet_bridge=bridge,
                    hl_client=hl_client,
                )

                if activated:
                    return {
                        "status": "active",
                        "mission_id": mission_id,
                        "message": "Mission activated (testnet auto-funded via Vault).",
                    }
                else:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Agent approval failed during testnet activation.",
                    )

            finally:
                await hl_client.close()
                await bridge.close()

        else:
            # =============================================
            # MAINNET PATH:
            # 0. Auto-detect source chain (Across bridge if not on Arbitrum)
            # 1. Transfer USDC from user's ZeroDev → Master EOA on Arbitrum
            # 2. Bridge Master EOA USDC → Hyperliquid L1 (Vault-signed)
            # 3. Deposit polling detects funds → Vault approves agent → ACTIVE
            # =============================================
            bridge = TurnkeyBridge()

            # Auto-detect which chain has funds (Arbitrum preferred = no bridge)
            wallet_id = mission.get("wallet_id", "")
            amount_str = str(mission["initial_capital"])
            source = await bridge.get_best_source_chain(
                wallet_id=wallet_id,
                amount=amount_str,
            )
            source_chain = source.get("chainId", 42161)
            needs_bridge = source.get("needsBridge", False)

            if needs_bridge and source_chain != 42161:
                # Cross-chain: bridge to Arbitrum via Across first
                logger.info(
                    "Cross-chain bridge needed for mission",
                    mission_id=mission_id,
                    source_chain=source_chain,
                )
                bridge_result = await bridge.across_bridge_to_arbitrum(
                    mission_id=mission_id,
                    wallet_id=wallet_id,
                    amount=amount_str,
                    source_chain_id=source_chain,
                    input_token=mission.get("deposit_currency", "USDC"),
                    recipient_address=master_eoa or "",
                )
                if not bridge_result.get("success", True):
                    await bridge.close()
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Cross-chain bridge failed: {bridge_result.get('error', 'Unknown')}",
                    )
                # Bridge poller will track PENDING → FILLED on wallet-service side
                # Continue to DEPOSITING status — deposit_polling will pick up from here

            if master_eoa:
                # Transfer USDC to Master EOA on Arbitrum, then bridge via Vault
                deposit_result = await bridge.transfer_usdc_to_master_eoa(
                    mission_id=mission_id,
                    master_eoa_address=master_eoa,
                    amount=amount_str,
                )
            else:
                # Fallback: legacy flow (direct bridge from smart wallet)
                deposit_result = await bridge.deposit_to_hyperliquid(
                    mission_id=mission_id,
                    amount=amount_str,
                )
            await bridge.close()

            if not deposit_result.get("success"):
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=deposit_result.get("error", "Deposit initiation failed"),
                )

            # Update status to DEPOSITING
            await update_mission_status(mission_id, "DEPOSITING")

            # Enqueue event-driven deposit check via arq
            try:
                from app.workers.deposit_worker import enqueue_deposit_check
                # Check balance on Master EOA (or fallback to user address)
                check_address = master_eoa or mission.get("kernel_account_address") or mission.get("user_wallet_address", "")
                await enqueue_deposit_check(
                    mission_id=mission_id,
                    expected_amount=float(mission["initial_capital"]),
                    user_address=check_address,
                    agent_address=mission.get("agent_address", ""),
                    initial_delay_seconds=settings.deposit_check_initial_delay_seconds,
                )
            except Exception as enq_err:
                logger.error(
                    "Failed to enqueue deposit check — reverting to PENDING",
                    error=str(enq_err),
                    mission_id=mission_id,
                )
                await update_mission_status(mission_id, "PENDING")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to start deposit monitoring. Please try again.",
                )

            return {
                "status": "depositing",
                "mission_id": mission_id,
                "message": "Deposit initiated. Mission will activate once funds arrive on Hyperliquid.",
                "deposit_tx": deposit_result.get("userOpHash"),
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Mission activation failed", error=str(e), mission_id=mission_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Mission activation failed",
        )


@router.post("/{mission_id}/pause")
async def pause_mission(
    mission_id: str,
    request: PauseMissionRequest,
    user: UserInfo = Depends(get_current_user),
):
    """
    Pause an active mission.

    This prevents the agent from opening new positions.
    Existing positions are NOT automatically closed.
    """
    from app.services.database import get_mission_by_id, update_mission_status

    logger.info("Pausing mission", mission_id=mission_id, user_id=user.user_id)

    mission = await get_mission_by_id(mission_id, user.user_id)
    if not mission:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Mission not found",
        )

    if mission["status"] != "ACTIVE":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Only ACTIVE missions can be paused. Current status: {mission['status']}",
        )

    await update_mission_status(mission_id, "PAUSED", {"pausedAt": datetime.utcnow()})

    return {
        "status": "paused",
        "mission_id": mission_id,
        "message": "Mission paused. No new trades will be opened. Existing positions remain open.",
    }


@router.post("/{mission_id}/resume")
async def resume_mission(
    mission_id: str,
    user: UserInfo = Depends(get_current_user),
):
    """Resume a paused mission."""
    from app.services.database import get_mission_by_id, update_mission_status

    logger.info("Resuming mission", mission_id=mission_id, user_id=user.user_id)

    mission = await get_mission_by_id(mission_id, user.user_id)
    if not mission:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Mission not found",
        )

    if mission["status"] != "PAUSED":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Only PAUSED missions can be resumed. Current status: {mission['status']}",
        )

    await update_mission_status(mission_id, "ACTIVE", {"pausedAt": None})

    return {
        "status": "active",
        "mission_id": mission_id,
        "message": "Mission resumed. Agent will trade on the next analysis cycle.",
    }


@router.post("/{mission_id}/revoke")
async def revoke_mission(
    mission_id: str,
    user: UserInfo = Depends(get_current_user),
):
    """
    Revoke agent access and end mission early.

    This will:
    1. Mark mission as REVOKED (positions will be force-closed by mission_lifecycle task)
    2. Funds will be returned to smart wallet after positions are closed
    """
    from app.services.database import get_mission_by_id, update_mission_status

    logger.info("Revoking mission", mission_id=mission_id, user_id=user.user_id)

    mission = await get_mission_by_id(mission_id, user.user_id)
    if not mission:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Mission not found",
        )

    revocable_statuses = {"PENDING", "ACTIVE", "PAUSED", "DEPOSITING", "APPROVING"}
    if mission["status"] not in revocable_statuses:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Mission cannot be revoked from '{mission['status']}' status.",
        )

    await update_mission_status(
        mission_id,
        "REVOKED",
        {"completedAt": datetime.utcnow()},
    )

    return {
        "status": "revoked",
        "mission_id": mission_id,
        "message": "Mission revoked. Open positions will be closed and funds returned.",
    }


def _get_max_leverage(risk_level: str) -> int:
    """Get maximum leverage for a risk level."""
    return {
        "CONSERVATIVE": 1,
        "MODERATE": 2,
        "AGGRESSIVE": 3,
    }.get(risk_level, 2)

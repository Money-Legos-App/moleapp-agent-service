"""
Mission Exit Orchestrator

Handles the complete exit flow when a mission ends (duration expired or user revokes):

Phase 3 of the 4-Pillar Architecture:
1. Halt & Flatten — revoke agent access, close all positions
2. Read Final Balance — get account value from Hyperliquid
3. Withdrawal — SDK withdraw_from_bridge() → USDC back to Arbitrum
3b. Wait for Settlement — poll Arbitrum USDC balance until withdrawal lands
4. Fee Collection & Payout — locally-signed Arbitrum TXs for fee + user return
5. Clean Up — NULL encrypted key ONLY after confirming Master EOA balance is zero
"""

import asyncio
from decimal import Decimal
from typing import Dict, Any, Optional

from eth_account import Account

import structlog

from app.services.vault.client import VaultEncryptionService
from app.services.vault.bridge_signer import ArbitrumBridgeSigner
from app.services.vault.payout_signer import calculate_fee_split, execute_fee_split

logger = structlog.get_logger(__name__)

# Hyperliquid API URLs
HL_MAINNET_API = "https://api.hyperliquid.xyz"
HL_TESTNET_API = "https://api.hyperliquid-testnet.xyz"

# Settlement polling config
SETTLEMENT_POLL_INTERVAL = 15  # seconds between balance checks
SETTLEMENT_MAX_WAIT = 600  # 10 minutes max wait for HL bridge settlement
SETTLEMENT_MIN_POLLS = 3  # minimum number of polls before giving up


async def complete_mission(
    mission_id: str,
    user_id: str,
    vault: VaultEncryptionService,
    bridge_signer: ArbitrumBridgeSigner,
    hl_client,  # HyperliquidClient (for reading balances/positions)
    wallet_bridge,  # TurnkeyBridge (for status updates)
    mission_data: Dict[str, Any],
    is_mainnet: bool = True,
) -> Dict[str, Any]:
    """
    Execute the full mission exit flow.

    Args:
        mission_id: The mission being completed
        user_id: The user who owns the mission
        vault: Vault encryption service for key decryption
        bridge_signer: Arbitrum TX signer
        hl_client: Hyperliquid API client
        wallet_bridge: Wallet service bridge for DB updates
        mission_data: Mission record from DB
        is_mainnet: True for mainnet, False for testnet

    Returns:
        Complete exit report with fee breakdown and TX hashes
    """
    master_eoa = mission_data.get("master_eoa_address", "")
    encrypted_key = mission_data.get("master_eoa_key_enc", "")
    initial_capital = Decimal(str(mission_data.get("initial_capital", 0)))
    user_wallet = mission_data.get("user_wallet_address", "")

    logger.info(
        "Starting mission exit",
        mission_id=mission_id,
        master_eoa=master_eoa,
        initial_capital=str(initial_capital),
    )

    result = {
        "mission_id": mission_id,
        "phase": "starting",
        "positions_closed": False,
        "withdrawal_initiated": False,
        "fee_split_completed": False,
        "vault_key_destroyed": False,
        "error": None,
    }

    try:
        # ─── Phase 1: Halt & Flatten ─────────────────────────
        result["phase"] = "closing_positions"
        await wallet_bridge.update_mission_status(mission_id, "COMPLETING")

        await _close_all_positions(
            mission_id, hl_client, mission_data,
            vault=vault, is_mainnet=is_mainnet,
        )
        result["positions_closed"] = True

        # ─── Phase 2: Read Final Balance ─────────────────────
        result["phase"] = "reading_balance"
        account_info = await hl_client.get_account_value(master_eoa)
        final_balance = Decimal(str(account_info.get("account_value", 0)))

        logger.info(
            "Final HL balance read",
            mission_id=mission_id,
            final_balance=str(final_balance),
        )

        # ─── Phase 3: Withdraw from HL to Arbitrum ───────────
        result["phase"] = "withdrawing"
        if final_balance > 0 and encrypted_key:
            await _withdraw_from_hyperliquid(
                vault=vault,
                encrypted_master_key=encrypted_key,
                amount=float(final_balance),
                master_eoa_address=master_eoa,
                is_mainnet=is_mainnet,
            )
            result["withdrawal_initiated"] = True

            logger.info(
                "HL withdrawal initiated — USDC returning to Arbitrum",
                mission_id=mission_id,
                amount=str(final_balance),
            )

        # ─── Phase 3b: Wait for HL Bridge Settlement ──────────
        # HL bridge withdrawals are ASYNC — USDC arrives on Arbitrum after
        # ~2-5 minutes. We MUST wait for funds to land before executing
        # the fee split, otherwise the Arbitrum TXs will fail.
        settled_balance = None
        if result["withdrawal_initiated"]:
            result["phase"] = "awaiting_settlement"
            settled_balance = await _wait_for_settlement(
                bridge_signer=bridge_signer,
                master_eoa_address=master_eoa,
                expected_amount=final_balance,
                mission_id=mission_id,
            )

            if settled_balance is None:
                raise RuntimeError(
                    f"HL withdrawal did not settle within {SETTLEMENT_MAX_WAIT}s. "
                    f"Funds may still be in transit. Key preserved for manual recovery."
                )

            logger.info(
                "HL withdrawal settled on Arbitrum",
                mission_id=mission_id,
                settled_usdc=str(settled_balance),
            )

        # ─── Phase 4: Fee Split on Arbitrum ──────────────────
        result["phase"] = "fee_split"
        from app.config import get_settings
        settings = get_settings()

        # Use fee percent locked at mission creation time (DB column),
        # falling back to current env var only for legacy missions
        locked_fee_percent = mission_data.get("feePercent")
        fee_percent = (
            float(locked_fee_percent)
            if locked_fee_percent is not None
            else settings.profit_fee_percent
        )

        # Use ACTUAL Arbitrum USDC balance for fee calc, not the HL L1 balance.
        # Bridge fees / rounding may cause the settled amount to differ slightly.
        if result["withdrawal_initiated"] and settled_balance is not None:
            actual_balance = settled_balance
        else:
            actual_balance = final_balance

        split = calculate_fee_split(
            initial_capital=initial_capital,
            final_balance=actual_balance,
            fee_percent=fee_percent,
        )

        if actual_balance > 0 and user_wallet and encrypted_key:
            fee_result = await execute_fee_split(
                mission_id=mission_id,
                bridge_signer=bridge_signer,
                encrypted_master_key=encrypted_key,
                master_eoa_address=master_eoa,
                treasury_address=settings.platform_treasury_address,
                user_wallet_address=user_wallet,
                initial_capital=initial_capital,
                final_balance=actual_balance,
                fee_percent=fee_percent,
            )
            result["fee_split_completed"] = True
            result["fee_tx_hash"] = fee_result.get("fee_tx_hash")
            result["payout_tx_hash"] = fee_result.get("payout_tx_hash")

        # ─── Phase 5: Finalize — NULL key ONLY after confirming balance is zero ──
        result["phase"] = "finalizing"

        from app.services.database import update_mission_status

        # Verify Master EOA USDC balance is zero before destroying key
        # This prevents irrecoverable fund loss if payout TX failed silently
        eoa_balance_zero = await _verify_eoa_balance_zero(
            bridge_signer=bridge_signer,
            master_eoa_address=master_eoa,
        )

        extra_fields = {
            "finalBalance": str(final_balance),
            "protocolFee": str(split["fee"]),
            "userPayout": str(split["user_payout"]),
        }

        if eoa_balance_zero:
            # Safe to NULL the key — no funds remain
            extra_fields["vaultKeyDestroyed"] = True
            extra_fields["masterEoaKeyEnc"] = None
            result["vault_key_destroyed"] = True
        else:
            # Funds still on Master EOA — preserve key for manual recovery
            logger.warning(
                "Master EOA still has USDC balance — preserving encrypted key",
                mission_id=mission_id,
                master_eoa=master_eoa,
            )
            extra_fields["vaultKeyDestroyed"] = False
            result["vault_key_destroyed"] = False

        await update_mission_status(
            mission_id=mission_id,
            new_status="COMPLETED",
            extra_fields=extra_fields,
        )

        # Also update via wallet_bridge for status sync
        await wallet_bridge.update_mission_status(
            mission_id=mission_id,
            status="COMPLETED",
            metadata={
                "finalBalance": str(final_balance),
                "protocolFee": str(split["fee"]),
                "userPayout": str(split["user_payout"]),
                "hadProfit": split["had_profit"],
                "feePercent": fee_percent,
                "keyPreserved": not eoa_balance_zero,
            },
        )

        result["phase"] = "completed"
        logger.info(
            "Mission exit completed",
            mission_id=mission_id,
            final_balance=str(final_balance),
            fee=str(split["fee"]),
            user_payout=str(split["user_payout"]),
        )

        # Audit: record mission exit
        from app.services.database import record_agent_audit
        await record_agent_audit(
            node="lifecycle",
            action="mission_exit_completed",
            mission_id=mission_id,
            user_id=user_id,
            metadata={
                "initial_capital": str(initial_capital),
                "final_balance": str(final_balance),
                "profit": str(split["profit"]),
                "fee": str(split["fee"]),
                "user_payout": str(split["user_payout"]),
                "had_profit": split["had_profit"],
                "withdrawal_initiated": result["withdrawal_initiated"],
                "fee_tx_hash": result.get("fee_tx_hash"),
                "payout_tx_hash": result.get("payout_tx_hash"),
            },
            success=True,
        )

    except Exception as e:
        logger.error(
            "Mission exit failed",
            mission_id=mission_id,
            phase=result["phase"],
            error=str(e),
        )
        result["error"] = str(e)

        # Audit: record failed exit
        try:
            from app.services.database import record_agent_audit
            await record_agent_audit(
                node="lifecycle",
                action="mission_exit_failed",
                mission_id=mission_id,
                user_id=user_id,
                error_message=str(e),
                metadata={"phase": result["phase"]},
                success=False,
            )
        except Exception:
            pass

    return result


async def _close_all_positions(
    mission_id: str,
    hl_client,
    mission_data: Dict[str, Any],
    vault: Optional[VaultEncryptionService] = None,
    is_mainnet: bool = True,
) -> None:
    """
    Close all open positions for this mission on Hyperliquid.

    Uses the agent key (not Master EOA) to place reduce-only market orders
    via the SDK. The agent key is Vault-decrypted just-in-time.
    """
    master_eoa = mission_data.get("master_eoa_address", "")
    agent_key_enc = mission_data.get("agent_private_key_enc", "")

    positions = await hl_client.get_positions(master_eoa)
    open_positions = [p for p in positions if float(p.get("quantity", 0)) != 0]

    if not open_positions:
        logger.info("No open positions to close", mission_id=mission_id)
        return

    # Need agent key to close positions (agent has trade permissions)
    if not agent_key_enc or not vault:
        logger.error(
            "Cannot close positions: no agent key or Vault unavailable",
            mission_id=mission_id,
            positions_count=len(open_positions),
        )
        raise RuntimeError("Agent key or Vault required to close positions")

    raw_agent_key = await vault.decrypt_private_key(agent_key_enc)
    try:
        agent_account = Account.from_key(raw_agent_key)

        from hyperliquid.exchange import Exchange

        base_url = HL_MAINNET_API if is_mainnet else HL_TESTNET_API
        exchange = Exchange(
            wallet=agent_account,
            base_url=base_url,
            account_address=master_eoa,
        )

        for pos in open_positions:
            asset = pos.get("asset", "")
            quantity = abs(float(pos.get("quantity", 0)))
            is_long = float(pos.get("quantity", 0)) > 0

            logger.info(
                "Closing position",
                mission_id=mission_id,
                asset=asset,
                quantity=quantity,
                direction="LONG" if is_long else "SHORT",
            )

            # Reduce-only market order in opposite direction
            result = exchange.market_close(coin=asset)

            logger.info(
                "Position close submitted",
                mission_id=mission_id,
                asset=asset,
                result_status=result.get("status") if isinstance(result, dict) else str(result),
            )

        # Audit: record position closing
        try:
            from app.services.database import record_agent_audit
            await record_agent_audit(
                node="lifecycle",
                action="positions_closed",
                mission_id=mission_id,
                metadata={
                    "positions_closed": len(open_positions),
                    "assets": [p.get("asset") for p in open_positions],
                },
                success=True,
            )
        except Exception:
            pass

    finally:
        del raw_agent_key
        try:
            del agent_account
        except NameError:
            pass
        try:
            del exchange
        except NameError:
            pass


async def _withdraw_from_hyperliquid(
    vault: VaultEncryptionService,
    encrypted_master_key: str,
    amount: float,
    master_eoa_address: str,
    is_mainnet: bool,
) -> Optional[Any]:
    """
    Withdraw USDC from Hyperliquid L1 back to Arbitrum using the SDK.

    Decrypts the Master EOA key just-in-time, creates an Exchange
    instance, and calls withdraw_from_bridge(). The SDK handles all
    EIP-712 signing internally.
    """
    raw_key = await vault.decrypt_private_key(encrypted_master_key)
    try:
        account = Account.from_key(raw_key)

        from hyperliquid.exchange import Exchange

        base_url = HL_MAINNET_API if is_mainnet else HL_TESTNET_API
        exchange = Exchange(wallet=account, base_url=base_url)

        result = exchange.withdraw_from_bridge(
            amount=amount,
            destination=master_eoa_address,
        )

        logger.info(
            "HL withdrawal submitted via SDK",
            master_eoa=master_eoa_address,
            amount=amount,
        )

        return result
    finally:
        del raw_key
        try:
            del account
        except NameError:
            pass
        try:
            del exchange
        except NameError:
            pass


async def _wait_for_settlement(
    bridge_signer: ArbitrumBridgeSigner,
    master_eoa_address: str,
    expected_amount: Decimal,
    mission_id: str,
) -> Optional[Decimal]:
    """
    Poll the Master EOA's USDC balance on Arbitrum until the HL bridge
    withdrawal lands.

    HL bridge withdrawals typically settle in 2-5 minutes. We poll every
    SETTLEMENT_POLL_INTERVAL seconds, up to SETTLEMENT_MAX_WAIT seconds.

    Returns:
        The USDC balance on Arbitrum (Decimal) if settlement detected, else None.
    """
    from eth_utils import to_checksum_address

    master = to_checksum_address(master_eoa_address)
    usdc_contract = bridge_signer.usdc

    # Read initial USDC balance before withdrawal arrives
    initial_balance = usdc_contract.functions.balanceOf(master).call()

    # We expect at least ~90% of the HL balance to arrive (some dust may remain)
    # Using 6 decimals for USDC
    expected_atomic = int(expected_amount * Decimal(10 ** 6))
    threshold = initial_balance + int(expected_atomic * 0.90)

    elapsed = 0
    polls = 0

    logger.info(
        "Waiting for HL bridge settlement",
        mission_id=mission_id,
        initial_usdc_balance=initial_balance,
        expected_atomic=expected_atomic,
        threshold=threshold,
    )

    while elapsed < SETTLEMENT_MAX_WAIT:
        await asyncio.sleep(SETTLEMENT_POLL_INTERVAL)
        elapsed += SETTLEMENT_POLL_INTERVAL
        polls += 1

        current_balance = usdc_contract.functions.balanceOf(master).call()

        logger.debug(
            "Settlement poll",
            mission_id=mission_id,
            poll=polls,
            current_balance=current_balance,
            threshold=threshold,
            elapsed_seconds=elapsed,
        )

        if current_balance >= threshold:
            settled_amount = Decimal(current_balance - initial_balance) / Decimal(10 ** 6)
            logger.info(
                "Bridge settlement detected",
                mission_id=mission_id,
                settled_usdc=str(settled_amount),
                polls=polls,
                elapsed_seconds=elapsed,
            )
            return settled_amount

    logger.error(
        "Bridge settlement timeout",
        mission_id=mission_id,
        elapsed_seconds=elapsed,
        polls=polls,
        last_balance=current_balance if polls > 0 else initial_balance,
    )
    return None


async def _verify_eoa_balance_zero(
    bridge_signer: ArbitrumBridgeSigner,
    master_eoa_address: str,
) -> bool:
    """
    Check that the Master EOA has no remaining USDC on Arbitrum.

    Only returns True if balance is zero (or dust < 0.01 USDC).
    This is the safety gate before NULLing the encrypted key.
    """
    from eth_utils import to_checksum_address

    master = to_checksum_address(master_eoa_address)
    usdc_balance = bridge_signer.usdc.functions.balanceOf(master).call()

    # Allow up to 0.01 USDC dust (10000 atomic units)
    DUST_THRESHOLD = 10_000

    if usdc_balance <= DUST_THRESHOLD:
        return True

    logger.warning(
        "Master EOA still has USDC",
        master_eoa=master,
        usdc_balance_atomic=usdc_balance,
        usdc_balance=float(usdc_balance) / 1e6,
    )
    return False

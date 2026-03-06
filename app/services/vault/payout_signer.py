"""
Payout Signer

Handles the exit fee split on Arbitrum after USDC is withdrawn from Hyperliquid:
1. TX1: Master EOA → Platform Treasury (30% of profit, if any)
2. TX2: Master EOA → User's ZeroDev Wallet (remaining balance)

Transactions are signed locally with the JIT-decrypted Master EOA key.
"""

from decimal import Decimal

import structlog

from app.services.vault.bridge_signer import ArbitrumBridgeSigner

logger = structlog.get_logger(__name__)

# USDC has 6 decimal places
USDC_DECIMALS = 6


def calculate_fee_split(
    initial_capital: Decimal,
    final_balance: Decimal,
    fee_percent: float = 30.0,
) -> dict:
    """
    Calculate the platform fee and user payout.

    Fee is only charged on PROFIT (positive yield).
    If the mission lost money, fee = 0 and user gets everything back.

    Args:
        initial_capital: Original deposit amount (e.g., Decimal("500"))
        final_balance: USDC balance at mission end (e.g., Decimal("600"))
        fee_percent: Platform fee percentage on profit (default: 30%)

    Returns:
        { profit, fee, user_payout, had_profit }
    """
    if not (0 <= fee_percent <= 50):
        raise ValueError(
            f"fee_percent must be between 0 and 50 (got {fee_percent}). "
            f"Check PROFIT_FEE_PERCENT configuration."
        )

    profit = final_balance - initial_capital

    if profit <= 0:
        # Loss or break-even: no fee
        return {
            "profit": profit,
            "fee": Decimal(0),
            "user_payout": final_balance,
            "had_profit": False,
        }

    fee = (profit * Decimal(str(fee_percent))) / Decimal("100")
    user_payout = final_balance - fee

    return {
        "profit": profit,
        "fee": fee,
        "user_payout": user_payout,
        "had_profit": True,
    }


def to_usdc_atomic(amount: Decimal) -> int:
    """Convert a USDC decimal amount to atomic units (6 decimals)."""
    return int(amount * Decimal(10 ** USDC_DECIMALS))


async def execute_fee_split(
    mission_id: str,
    bridge_signer: ArbitrumBridgeSigner,
    encrypted_master_key: str,
    master_eoa_address: str,
    treasury_address: str,
    user_wallet_address: str,
    initial_capital: Decimal,
    final_balance: Decimal,
    fee_percent: float = 30.0,
) -> dict:
    """
    Execute the full fee split on Arbitrum after HL withdrawal.

    Steps:
    1. Calculate the 30% profit fee
    2. TX1: Send fee to Platform Treasury (if profit > 0)
    3. TX2: Send remaining balance to User's ZeroDev Wallet

    Returns:
        { fee_amount, user_payout, fee_tx_hash, payout_tx_hash }
    """
    split = calculate_fee_split(initial_capital, final_balance, fee_percent)

    logger.info(
        "Executing fee split",
        mission_id=mission_id,
        initial_capital=str(initial_capital),
        final_balance=str(final_balance),
        profit=str(split["profit"]),
        fee=str(split["fee"]),
        user_payout=str(split["user_payout"]),
        had_profit=split["had_profit"],
    )

    result = {
        "fee_amount": split["fee"],
        "user_payout": split["user_payout"],
        "fee_tx_hash": None,
        "payout_tx_hash": None,
    }

    # TX1: Fee → Platform Treasury (only if there was profit)
    if split["had_profit"] and split["fee"] > 0:
        fee_atomic = to_usdc_atomic(split["fee"])
        result["fee_tx_hash"] = await bridge_signer.transfer_usdc(
            encrypted_master_key=encrypted_master_key,
            master_eoa_address=master_eoa_address,
            to_address=treasury_address,
            usdc_amount=fee_atomic,
        )
        logger.info(
            "Fee sent to treasury",
            mission_id=mission_id,
            fee=str(split["fee"]),
            tx_hash=result["fee_tx_hash"],
        )

    # TX2: Payout → User's ZeroDev Wallet
    payout_atomic = to_usdc_atomic(split["user_payout"])
    if payout_atomic > 0:
        result["payout_tx_hash"] = await bridge_signer.transfer_usdc(
            encrypted_master_key=encrypted_master_key,
            master_eoa_address=master_eoa_address,
            to_address=user_wallet_address,
            usdc_amount=payout_atomic,
        )
        logger.info(
            "Payout sent to user",
            mission_id=mission_id,
            payout=str(split["user_payout"]),
            tx_hash=result["payout_tx_hash"],
        )

    logger.info("Fee split complete", mission_id=mission_id)

    return result

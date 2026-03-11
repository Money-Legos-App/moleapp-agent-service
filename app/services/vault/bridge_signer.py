"""
Arbitrum Signer (Fee/Payout Operations)

Builds and signs Arbitrum transactions for:
- USDC.transfer(destination, amount) — transfer USDC on Arbitrum (fee/payout)

Bridge operations (approve+sendUsd) have been removed — Across Protocol now
bridges USDC directly to HyperEVM (chain 999), which auto-routes to HyperCore.

Private keys are decrypted from Vault just-in-time, used for local signing
via eth_account, then immediately dropped from memory.
"""

from typing import Optional

from eth_account import Account
from eth_utils import to_checksum_address
from web3 import Web3

import structlog

from app.services.vault.client import VaultEncryptionService

logger = structlog.get_logger(__name__)

# Contract ABIs (minimal, only functions we call)
ERC20_ABI = [
    {
        "name": "approve",
        "type": "function",
        "inputs": [
            {"name": "spender", "type": "address"},
            {"name": "amount", "type": "uint256"},
        ],
        "outputs": [{"name": "", "type": "bool"}],
    },
    {
        "name": "transfer",
        "type": "function",
        "inputs": [
            {"name": "to", "type": "address"},
            {"name": "amount", "type": "uint256"},
        ],
        "outputs": [{"name": "", "type": "bool"}],
    },
]

class ArbitrumBridgeSigner:
    """Signs Arbitrum transactions using locally-decrypted Vault keys.

    Used for fee collection and payout transfers on Arbitrum.
    Bridge operations are handled by Across Protocol (direct to HyperEVM).
    """

    def __init__(
        self,
        vault: VaultEncryptionService,
        rpc_url: str,
        usdc_address: str,
        chain_id: int = 42161,  # Arbitrum One
    ):
        self.vault = vault
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        self.chain_id = chain_id
        self.usdc = self.w3.eth.contract(
            address=to_checksum_address(usdc_address),
            abi=ERC20_ABI,
        )

    async def transfer_usdc(
        self,
        encrypted_master_key: str,
        master_eoa_address: str,
        to_address: str,
        usdc_amount: int,
        nonce_override: int = None,
    ) -> dict:
        """
        Transfer USDC from Master EOA to any address on Arbitrum.
        Used for fee collection (→ Treasury) and user payout (→ ZeroDev wallet).

        Returns:
            { tx_hash, nonce, success }
        """
        master = to_checksum_address(master_eoa_address)
        to = to_checksum_address(to_address)

        result = await self._sign_and_send(
            encrypted_master_key=encrypted_master_key,
            master_eoa=master,
            contract_call=self.usdc.functions.transfer(to, usdc_amount),
            nonce_override=nonce_override,
        )

        logger.info(
            "USDC transfer sent",
            from_addr=master,
            to_addr=to,
            amount=usdc_amount,
            tx_hash=result["tx_hash"],
            success=result["success"],
        )

        return result

    async def _sign_and_send(
        self,
        encrypted_master_key: str,
        master_eoa: str,
        contract_call,
        nonce_override: int = None,
    ) -> dict:
        """
        Decrypt key → sign locally → broadcast → wait for receipt.

        1. Decrypt Master EOA key from Vault (JIT)
        2. Build unsigned TX from contract call
        3. Sign locally with eth_account
        4. Broadcast the signed TX
        5. Wait for receipt and verify success
        6. Delete raw key from memory

        Returns:
            { tx_hash, nonce, success, receipt }
        """
        raw_key = await self.vault.decrypt_private_key(encrypted_master_key)
        try:
            account = Account.from_key(raw_key)

            nonce = nonce_override if nonce_override is not None else self.w3.eth.get_transaction_count(master_eoa)
            tx = contract_call.build_transaction({
                "from": master_eoa,
                "nonce": nonce,
                "gas": 100_000,  # Sufficient for ERC20 approve/transfer
                "gasPrice": self.w3.eth.gas_price,
                "chainId": self.chain_id,
            })

            signed = account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)

            # Wait for receipt and verify TX succeeded on-chain
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            success = receipt.status == 1

            if not success:
                logger.error(
                    "TX reverted on-chain",
                    tx_hash=tx_hash.hex(),
                    nonce=nonce,
                    gas_used=receipt.gasUsed,
                )

            return {
                "tx_hash": tx_hash.hex(),
                "nonce": nonce,
                "success": success,
            }
        finally:
            del raw_key
            try:
                del account
            except NameError:
                pass
            try:
                del signed
            except NameError:
                pass

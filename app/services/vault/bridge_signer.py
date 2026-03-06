"""
Arbitrum Bridge Signer

Builds and signs Arbitrum transactions for:
1. USDC.approve(HL_BRIDGE, amount) — approve bridge to spend USDC
2. HLBridge.sendUsd(masterEoa, amount) — bridge USDC to Hyperliquid L1
3. USDC.transfer(destination, amount) — transfer USDC on Arbitrum (fee/payout)

Private keys are decrypted from Vault just-in-time, used for local signing
via eth_account, then immediately dropped from memory.

The Gas Station Wallet provides ETH for gas.
"""

from typing import Optional

from eth_account import Account
from eth_utils import to_checksum_address
from web3 import Web3
from web3.types import TxParams

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

HL_BRIDGE_ABI = [
    {
        "name": "sendUsd",
        "type": "function",
        "inputs": [
            {"name": "destination", "type": "address"},
            {"name": "amount", "type": "uint64"},
        ],
        "outputs": [],
    },
]


class ArbitrumBridgeSigner:
    """Builds and signs Arbitrum transactions using locally-decrypted keys."""

    def __init__(
        self,
        vault: VaultEncryptionService,
        rpc_url: str,
        usdc_address: str,
        hl_bridge_address: str,
        gas_station_key: str,
        chain_id: int = 42161,  # Arbitrum One
    ):
        self.vault = vault
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        self.chain_id = chain_id
        self.usdc = self.w3.eth.contract(
            address=to_checksum_address(usdc_address),
            abi=ERC20_ABI,
        )
        self.hl_bridge = self.w3.eth.contract(
            address=to_checksum_address(hl_bridge_address),
            abi=HL_BRIDGE_ABI,
        )
        # Encrypt gas station key via Vault at init; raw key never persists in memory.
        # _gas_station_key_enc holds the Vault ciphertext (or None if not configured).
        self._gas_station_key_enc: Optional[str] = None
        self._gas_station_address: Optional[str] = None
        if gas_station_key:
            temp = Account.from_key(gas_station_key)
            self._gas_station_address = temp.address
            del temp
            # Vault encryption is async — store raw key temporarily; call
            # seal_gas_station_key() before first use to encrypt and wipe it.
            self.__gas_station_key_pending: Optional[str] = gas_station_key
        else:
            self.__gas_station_key_pending = None

    async def seal_gas_station_key(self) -> None:
        """Encrypt the Gas Station private key via Vault Transit and clear the raw copy.

        Must be called once after construction (before any fund_gas / bridge calls).
        Safe to call multiple times — no-ops after the first successful seal.
        """
        if self.__gas_station_key_pending is None:
            return  # already sealed or not configured
        try:
            self._gas_station_key_enc = await self.vault.encrypt_private_key(
                self.__gas_station_key_pending
            )
            logger.info("Gas Station key sealed via Vault Transit")
        finally:
            # Wipe raw key regardless of success/failure
            self.__gas_station_key_pending = None

    async def fund_gas(self, master_eoa_address: str, amount_wei: int = 100_000_000_000_000) -> Optional[str]:
        """
        Send ETH from Gas Station to Master EOA for gas.

        Default: 0.0001 ETH (~$0.01, enough for ~3-5 Arbitrum TXs).
        The gas station key is decrypted just-in-time from Vault and cleared after use.
        """
        # Ensure the raw key has been sealed into Vault on first call
        await self.seal_gas_station_key()

        if not self._gas_station_key_enc:
            logger.warning("No gas station configured, skipping gas funding")
            return None

        master = to_checksum_address(master_eoa_address)
        balance = self.w3.eth.get_balance(master)

        if balance >= amount_wei:
            logger.debug("Master EOA already has gas", address=master, balance=balance)
            return None

        # JIT decrypt Gas Station key from Vault — same pattern as Master EOA
        raw_key = await self.vault.decrypt_private_key(self._gas_station_key_enc)
        try:
            gas_station = Account.from_key(raw_key)

            nonce = self.w3.eth.get_transaction_count(gas_station.address)
            tx: TxParams = {
                "to": master,
                "value": amount_wei,
                "gas": 21_000,
                "gasPrice": self.w3.eth.gas_price,
                "nonce": nonce,
                "chainId": self.chain_id,
            }

            signed = gas_station.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)

            logger.info(
                "Gas station funded Master EOA",
                master=master,
                amount_wei=amount_wei,
                tx_hash=tx_hash.hex(),
            )

            return tx_hash.hex()
        finally:
            del raw_key
            try:
                del gas_station
            except NameError:
                pass
            try:
                del signed
            except NameError:
                pass

    async def bridge_to_hyperliquid(
        self,
        encrypted_master_key: str,
        master_eoa_address: str,
        usdc_amount: int,  # USDC atomic units (6 decimals)
    ) -> dict:
        """
        Bridge USDC from Master EOA on Arbitrum to Hyperliquid L1.

        Steps:
        1. Fund Master EOA with gas (from Gas Station)
        2. USDC.approve(HL_BRIDGE, amount) — locally signed
        3. HLBridge.sendUsd(masterEoa, amount) — locally signed

        Returns { approve_tx, bridge_tx, success }
        """
        master = to_checksum_address(master_eoa_address)

        logger.info(
            "Bridging USDC to Hyperliquid",
            master_eoa=master,
            usdc_amount=usdc_amount,
        )

        # Step 1: Ensure Master EOA has gas
        await self.fund_gas(master)

        # Step 2: USDC.approve(HL_BRIDGE, amount)
        approve_tx_hash = await self._sign_and_send(
            encrypted_master_key=encrypted_master_key,
            master_eoa=master,
            contract_call=self.usdc.functions.approve(
                self.hl_bridge.address, usdc_amount
            ),
        )

        logger.info("USDC approve TX sent", tx_hash=approve_tx_hash)

        # Step 3: HLBridge.sendUsd(masterEoa, amount)
        bridge_tx_hash = await self._sign_and_send(
            encrypted_master_key=encrypted_master_key,
            master_eoa=master,
            contract_call=self.hl_bridge.functions.sendUsd(master, usdc_amount),
        )

        logger.info("Bridge TX sent", tx_hash=bridge_tx_hash)

        # Audit: record bridge operation
        try:
            from app.services.database import record_agent_audit
            await record_agent_audit(
                node="lifecycle",
                action="usdc_bridged_to_hyperliquid",
                metadata={
                    "master_eoa": master,
                    "usdc_amount": usdc_amount,
                    "approve_tx": approve_tx_hash,
                    "bridge_tx": bridge_tx_hash,
                },
                success=True,
            )
        except Exception:
            pass

        return {
            "approve_tx": approve_tx_hash,
            "bridge_tx": bridge_tx_hash,
            "success": True,
        }

    async def transfer_usdc(
        self,
        encrypted_master_key: str,
        master_eoa_address: str,
        to_address: str,
        usdc_amount: int,
    ) -> str:
        """
        Transfer USDC from Master EOA to any address on Arbitrum.
        Used for fee collection (→ Treasury) and user payout (→ ZeroDev wallet).
        """
        master = to_checksum_address(master_eoa_address)
        to = to_checksum_address(to_address)

        await self.fund_gas(master)

        tx_hash = await self._sign_and_send(
            encrypted_master_key=encrypted_master_key,
            master_eoa=master,
            contract_call=self.usdc.functions.transfer(to, usdc_amount),
        )

        logger.info(
            "USDC transfer sent",
            from_addr=master,
            to_addr=to,
            amount=usdc_amount,
            tx_hash=tx_hash,
        )

        return tx_hash

    async def _sign_and_send(
        self,
        encrypted_master_key: str,
        master_eoa: str,
        contract_call,
    ) -> str:
        """
        Decrypt key → sign locally → broadcast.

        1. Decrypt Master EOA key from Vault (JIT)
        2. Build unsigned TX from contract call
        3. Sign locally with eth_account
        4. Broadcast the signed TX
        5. Delete raw key from memory
        """
        raw_key = await self.vault.decrypt_private_key(encrypted_master_key)
        try:
            account = Account.from_key(raw_key)

            nonce = self.w3.eth.get_transaction_count(master_eoa)
            tx = contract_call.build_transaction({
                "from": master_eoa,
                "nonce": nonce,
                "gas": 100_000,  # Sufficient for ERC20 approve/transfer
                "gasPrice": self.w3.eth.gas_price,
                "chainId": self.chain_id,
            })

            signed = account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)

            return tx_hash.hex()
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

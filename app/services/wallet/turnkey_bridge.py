"""
Turnkey Bridge Service
Handles communication with wallet-service for Turnkey signing operations
"""

from typing import Any, Dict, List, Optional

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

logger = structlog.get_logger(__name__)


def _should_retry(retry_state) -> bool:
    """Don't retry on client errors (4xx) — only retry on server/network errors."""
    exc = retry_state.outcome.exception()
    if isinstance(exc, httpx.HTTPStatusError) and 400 <= exc.response.status_code < 500:
        return False
    return exc is not None


class TurnkeyBridge:
    """
    Bridge to wallet-service for Turnkey signing operations.

    The agent-service (Python) needs to request signatures from
    wallet-service (TypeScript) which has access to Turnkey SDK.

    Authentication: Keycloak service token (Client Credentials flow)

    Flow:
    1. Agent builds order payload
    2. Agent sends payload to wallet-service (with Keycloak auth)
    3. Wallet-service validates mission and signs via Turnkey
    4. Wallet-service returns signed payload
    5. Agent submits to Hyperliquid
    """

    def __init__(
        self,
        wallet_service_url: Optional[str] = None,
    ):
        """
        Initialize the Turnkey bridge.

        Args:
            wallet_service_url: Wallet service base URL
        """
        from app.config import get_settings

        settings = get_settings()
        self.wallet_service_url = wallet_service_url or settings.wallet_service_url

        if not settings.keycloak_client_secret:
            raise ValueError("KEYCLOAK_CLIENT_SECRET is required for service-to-service auth")

        self._client: Optional[httpx.AsyncClient] = None
        self._keycloak_auth = None

        logger.info(
            "TurnkeyBridge initialized",
            wallet_service_url=self.wallet_service_url,
        )

    async def _get_keycloak_auth(self):
        """Get or create the Keycloak auth client."""
        if self._keycloak_auth is None:
            from app.services.auth import get_keycloak_auth
            self._keycloak_auth = get_keycloak_auth()
        return self._keycloak_auth

    async def _get_auth_headers(self) -> Dict[str, str]:
        """Get Keycloak authentication headers."""
        headers = {"Content-Type": "application/json"}

        keycloak = await self._get_keycloak_auth()
        auth_headers = await keycloak.get_auth_headers()
        headers.update(auth_headers)

        return headers

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.wallet_service_url,
                timeout=60.0,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client and cleanup resources."""
        if self._client:
            await self._client.aclose()
            self._client = None
        if self._keycloak_auth:
            await self._keycloak_auth.close()
            self._keycloak_auth = None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=lambda retry_state: _should_retry(retry_state),
    )
    async def _request(
        self,
        method: str,
        path: str,
        json: Dict = None,
    ) -> Dict[str, Any]:
        """Make an authenticated request to wallet-service."""
        client = await self._get_client()
        headers = await self._get_auth_headers()

        response = await client.request(
            method=method,
            url=path,
            json=json,
            headers=headers,
        )

        # For client errors (4xx), return the error body instead of raising
        # so callers can inspect error codes (e.g. FORBIDDEN, NOT_FOUND)
        if 400 <= response.status_code < 500:
            try:
                body = response.json()
            except Exception:
                body = {"error": f"HTTP {response.status_code}", "message": response.text}
            body["success"] = False
            return body

        response.raise_for_status()
        return response.json()

    # ==================
    # Mission Management
    # ==================

    async def create_mission(
        self,
        user_id: str,
        mission_type: str,
        deposit_amount: str,
        wallet_id: str,
    ) -> Dict[str, Any]:
        """
        Create a new agent mission.

        Args:
            user_id: User's ID
            mission_type: "SHORT_TERM_30D" or "LONG_TERM_45D"
            deposit_amount: Initial deposit in USDC
            wallet_id: User's wallet ID

        Returns:
            Mission creation result including mission ID and user wallet address
        """
        result = await self._request(
            "POST",
            "/internal/v1/agent/create-mission",
            json={
                "userId": user_id,
                "missionType": mission_type,
                "depositAmount": deposit_amount,
                "walletId": wallet_id,
            },
        )

        logger.info(
            "Mission created via wallet-service",
            mission_id=result.get("missionId"),
            user_id=user_id,
        )

        return result

    async def validate_mission(self, mission_id: str) -> Dict[str, Any]:
        """
        Validate a mission and its agent approval status.

        Args:
            mission_id: Mission ID to validate

        Returns:
            Validation result with mission details
        """
        result = await self._request(
            "GET",
            f"/internal/v1/agent/validate-mission/{mission_id}",
        )

        return result

    # ==================
    # Trade Signing
    # ==================

    async def sign_trade(
        self,
        mission_id: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Request a trade signature from wallet-service (legacy format).

        Args:
            mission_id: Mission ID
            payload: Hyperliquid order payload to sign

        Returns:
            Signed payload ready for submission to Hyperliquid
        """
        try:
            result = await self._request(
                "POST",
                "/internal/v1/agent/sign-trade",
                json={
                    "missionId": mission_id,
                    "payload": payload,
                },
            )

            logger.info(
                "Trade signed via wallet-service",
                mission_id=mission_id,
                success=result.get("success", False),
            )

            return {
                "success": result.get("success", False),
                "signature": result.get("signature"),
                "signedPayload": result.get("signedPayload"),
                "action": payload,
                "nonce": result.get("nonce"),
            }

        except httpx.HTTPStatusError as e:
            logger.error(
                "Trade signing failed",
                mission_id=mission_id,
                status_code=e.response.status_code,
                error=str(e),
            )
            return {
                "success": False,
                "error": f"HTTP {e.response.status_code}: {e.response.text}",
            }
        except Exception as e:
            logger.error(
                "Trade signing exception",
                mission_id=mission_id,
                error=str(e),
            )
            return {
                "success": False,
                "error": str(e),
            }

    async def sign_typed_data(
        self,
        mission_id: str,
        typed_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Request EIP-712 typed data signature from wallet-service.

        This is the correct method for Hyperliquid order signing.
        Uses Turnkey's signTypedData instead of signMessage.

        Args:
            mission_id: Mission ID
            typed_data: Full EIP-712 structure with domain, types, primaryType, message

        Returns:
            Signature and original typed data
        """
        try:
            result = await self._request(
                "POST",
                "/internal/v1/agent/sign-typed-data",
                json={
                    "missionId": mission_id,
                    "typedData": {
                        "domain": typed_data.get("domain"),
                        "types": typed_data.get("types"),
                        "primaryType": typed_data.get("primaryType"),
                        "message": typed_data.get("message"),
                    },
                },
            )

            logger.info(
                "EIP-712 data signed via wallet-service",
                mission_id=mission_id,
                success=result.get("success", False),
                primary_type=typed_data.get("primaryType"),
            )

            return {
                "success": result.get("success", False),
                "signature": result.get("signature"),
                "typedData": typed_data,
                "nonce": typed_data.get("message", {}).get("nonce"),
                "error": result.get("error"),
            }

        except httpx.HTTPStatusError as e:
            logger.error(
                "EIP-712 signing failed",
                mission_id=mission_id,
                status_code=e.response.status_code,
                error=str(e),
            )
            return {
                "success": False,
                "error": f"HTTP {e.response.status_code}: {e.response.text}",
            }
        except Exception as e:
            logger.error(
                "EIP-712 signing exception",
                mission_id=mission_id,
                error=str(e),
            )
            return {
                "success": False,
                "error": str(e),
            }

    async def batch_sign_typed_data(
        self,
        orders: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Request batch EIP-712 signatures for multiple orders.

        Args:
            orders: List of {mission_id, typed_data} dicts

        Returns:
            List of signing results in the same order
        """
        if not orders:
            return []

        try:
            result = await self._request(
                "POST",
                "/internal/v1/agent/batch-sign-typed-data",
                json={
                    "orders": [
                        {
                            "missionId": order["mission_id"],
                            "typedData": {
                                "domain": order["typed_data"].get("domain"),
                                "types": order["typed_data"].get("types"),
                                "primaryType": order["typed_data"].get("primaryType"),
                                "message": order["typed_data"].get("message"),
                            },
                        }
                        for order in orders
                    ]
                },
            )

            results = result.get("results", [])

            success_count = sum(1 for r in results if r.get("success"))
            logger.info(
                "Batch EIP-712 signing completed",
                total=len(orders),
                successful=success_count,
                failed=len(orders) - success_count,
            )

            return [
                {
                    "success": r.get("success", False),
                    "signature": r.get("signature"),
                    "typedData": orders[i]["typed_data"] if i < len(orders) else None,
                    "nonce": orders[i]["typed_data"].get("message", {}).get("nonce") if i < len(orders) else None,
                    "error": r.get("error"),
                }
                for i, r in enumerate(results)
            ]

        except Exception as e:
            logger.error(
                "Batch EIP-712 signing failed",
                orders_count=len(orders),
                error=str(e),
            )
            return [
                {"success": False, "error": str(e)}
                for _ in orders
            ]

    async def batch_sign_trades(
        self,
        orders: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Request batch signatures for multiple orders.

        This is more efficient than signing one at a time when
        processing multiple missions in a single analysis cycle.

        Args:
            orders: List of {mission_id, payload} dicts

        Returns:
            List of signing results in the same order
        """
        if not orders:
            return []

        try:
            result = await self._request(
                "POST",
                "/internal/v1/agent/batch-sign",
                json={"orders": orders},
            )

            results = result.get("results", [])

            success_count = sum(1 for r in results if r.get("success"))
            logger.info(
                "Batch signing completed",
                total=len(orders),
                successful=success_count,
                failed=len(orders) - success_count,
            )

            # Ensure we return results in the same order as input
            return [
                {
                    "success": r.get("success", False),
                    "signature": r.get("signature"),
                    "signedPayload": r.get("signedPayload"),
                    "action": orders[i]["payload"] if i < len(orders) else None,
                    "nonce": r.get("nonce"),
                    "error": r.get("error"),
                }
                for i, r in enumerate(results)
            ]

        except Exception as e:
            logger.error(
                "Batch signing failed",
                orders_count=len(orders),
                error=str(e),
            )
            # Return failures for all orders
            return [
                {"success": False, "error": str(e)}
                for _ in orders
            ]

    # ==================
    # Agent Approval
    # ==================

    async def sign_agent_approval(
        self,
        mission_id: str,
        agent_address: str,
        typed_data: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Request signature for agent approval transaction.

        Agent-service pre-computes the phantom agent EIP-712 typed data
        (including the action hash / connectionId). Wallet-service signs
        it with the user's master EOA via Turnkey.

        Args:
            mission_id: Mission ID
            agent_address: MoleApp's agent wallet address
            typed_data: Pre-computed EIP-712 typed data from HL client

        Returns:
            Dict with 'success' and 'signature' (hex string)
        """
        try:
            result = await self._request(
                "POST",
                "/internal/v1/agent/sign-approval",
                json={
                    "missionId": mission_id,
                    "agentAddress": agent_address,
                    "typedData": typed_data,
                },
            )

            logger.info(
                "Agent approval signed",
                mission_id=mission_id,
                success=result.get("success", False),
            )

            return result

        except Exception as e:
            logger.error(
                "Agent approval signing failed",
                mission_id=mission_id,
                error=str(e),
            )
            return {
                "success": False,
                "error": str(e),
            }

    # ==================
    # Account Operations
    # ==================

    async def get_mission_details(self, mission_id: str) -> Dict[str, Any]:
        """
        Get full mission details including user wallet info.

        Args:
            mission_id: Mission ID

        Returns:
            Mission details with wallet information
        """
        result = await self._request(
            "GET",
            f"/internal/v1/agent/mission/{mission_id}",
        )
        return result

    async def get_user_wallet_address(self, mission_id: str) -> Optional[str]:
        """
        Get the user's wallet address for a mission.

        Args:
            mission_id: Mission ID

        Returns:
            User's wallet address or None
        """
        try:
            details = await self.get_mission_details(mission_id)
            return details.get("userWalletAddress")
        except Exception as e:
            logger.error(
                "Failed to get user wallet address",
                mission_id=mission_id,
                error=str(e),
            )
            return None

    async def update_mission_status(
        self,
        mission_id: str,
        status: str,
        metadata: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Update mission status.

        Args:
            mission_id: Mission ID
            status: New status (ACTIVE, PAUSED, COMPLETED, etc.)
            metadata: Additional metadata to update

        Returns:
            Update result
        """
        result = await self._request(
            "PATCH",
            f"/internal/v1/agent/mission/{mission_id}/status",
            json={
                "status": status,
                "metadata": metadata or {},
            },
        )

        logger.info(
            "Mission status updated",
            mission_id=mission_id,
            status=status,
        )

        return result

    async def record_pnl_snapshot(
        self,
        mission_id: str,
        total_value: float,
        total_pnl: float,
        unrealized_pnl: float,
        realized_pnl: float,
    ) -> Dict[str, Any]:
        """
        Record a PnL snapshot for a mission.

        Args:
            mission_id: Mission ID
            total_value: Current total value
            total_pnl: Total PnL
            unrealized_pnl: Unrealized PnL
            realized_pnl: Realized PnL

        Returns:
            Snapshot creation result
        """
        result = await self._request(
            "POST",
            f"/internal/v1/agent/mission/{mission_id}/pnl-snapshot",
            json={
                "totalValue": str(total_value),
                "totalPnl": str(total_pnl),
                "unrealizedPnl": str(unrealized_pnl),
                "realizedPnl": str(realized_pnl),
            },
        )

        return result

    # ==================
    # Hyperliquid Bridge
    # ==================

    async def deposit_to_hyperliquid(
        self,
        mission_id: str,
        amount: str,
        chain_id: int = None,
    ) -> Dict[str, Any]:
        """
        Initiate a USDC deposit to Hyperliquid via the bridge on Arbitrum.

        This triggers a gasless UserOperation that:
        1. Approves USDC spending by the bridge contract
        2. Deposits USDC to the Hyperliquid bridge

        The mission status transitions: PENDING → DEPOSITING.

        Args:
            mission_id: Mission ID
            amount: USDC amount as string (e.g., "10.0")
            chain_id: Target chain ID (defaults to mainnet/testnet based on config)

        Returns:
            Deposit result with userOpHash for tracking
        """
        if chain_id is None:
            from app.config import get_settings
            settings = get_settings()
            chain_id = 42161 if settings.hyperliquid_mainnet else 421614
        try:
            result = await self._request(
                "POST",
                "/internal/v1/agent/deposit-to-hyperliquid",
                json={
                    "missionId": mission_id,
                    "amount": amount,
                    "chainId": chain_id,
                },
            )

            logger.info(
                "HL bridge deposit initiated via wallet-service",
                mission_id=mission_id,
                user_op_hash=result.get("userOpHash"),
                success=result.get("success", False),
            )

            return result

        except Exception as e:
            logger.error(
                "HL bridge deposit failed",
                mission_id=mission_id,
                error=str(e),
            )
            return {
                "success": False,
                "error": str(e),
            }

    async def withdraw_from_hyperliquid(
        self,
        mission_id: str,
        amount: str,
    ) -> Dict[str, Any]:
        """
        Sign a withdrawal from Hyperliquid back to the user's smart wallet.

        IMPORTANT: On Hyperliquid, only the MASTER address (user's Turnkey EOA)
        can sign withdrawals. The agent wallet cannot withdraw.

        This calls wallet-service which signs the withdrawal request using
        the user's master EOA via Turnkey.

        Args:
            mission_id: Mission ID
            amount: USDC amount to withdraw

        Returns:
            Signed withdrawal payload ready for submission to Hyperliquid
        """
        try:
            result = await self._request(
                "POST",
                "/internal/v1/agent/withdraw-from-hyperliquid",
                json={
                    "missionId": mission_id,
                    "amount": amount,
                },
            )

            logger.info(
                "HL withdrawal signed via wallet-service",
                mission_id=mission_id,
                success=result.get("success", False),
            )

            return result

        except Exception as e:
            logger.error(
                "HL withdrawal signing failed",
                mission_id=mission_id,
                error=str(e),
            )
            return {
                "success": False,
                "error": str(e),
            }

    # ==================
    # Vault Master EOA Support
    # ==================

    async def transfer_usdc_to_master_eoa(
        self,
        mission_id: str,
        master_eoa_address: str,
        amount: str,
    ) -> Dict[str, Any]:
        """
        Transfer USDC from user's ZeroDev wallet to the mission's Master EOA on Arbitrum.

        Executed as a gasless UserOperation via Pimlico paymaster.

        Args:
            mission_id: Mission ID
            master_eoa_address: Vault-derived Master EOA address on Arbitrum
            amount: USDC amount as string (e.g., "500.0")

        Returns:
            Result with userOpHash for tracking
        """
        try:
            result = await self._request(
                "POST",
                "/internal/v1/agent/transfer-to-master-eoa",
                json={
                    "missionId": mission_id,
                    "masterEoaAddress": master_eoa_address,
                    "amount": amount,
                },
            )

            logger.info(
                "USDC transfer to Master EOA initiated",
                mission_id=mission_id,
                master_eoa=master_eoa_address,
                user_op_hash=result.get("userOpHash"),
                success=result.get("success", False),
            )

            return result

        except Exception as e:
            logger.error(
                "Transfer to Master EOA failed",
                mission_id=mission_id,
                master_eoa=master_eoa_address,
                error=str(e),
            )
            return {
                "success": False,
                "error": str(e),
            }

    async def store_agent_address(
        self,
        mission_id: str,
        agent_address: str,
    ) -> Dict[str, Any]:
        """
        Inform wallet-service of the agent address for a mission.

        The agent key itself is stored as Vault ciphertext in the DB by
        agent-service — it never leaves this process over HTTP.

        Args:
            mission_id: Mission ID
            agent_address: Agent's Ethereum address

        Returns:
            Result with success status
        """
        try:
            result = await self._request(
                "POST",
                f"/internal/v1/agent/mission/{mission_id}/store-agent-key",
                json={
                    "agentAddress": agent_address,
                },
            )

            logger.info(
                "Agent address stored via wallet-service",
                mission_id=mission_id,
                agent_address=agent_address,
            )

            return result

        except Exception as e:
            logger.error(
                "Failed to store agent address",
                mission_id=mission_id,
                error=str(e),
            )
            return {
                "success": False,
                "error": str(e),
            }

    # ==================
    # Across Bridge (Cross-Chain to Arbitrum)
    # ==================

    async def get_best_source_chain(
        self,
        wallet_id: str,
        amount: str,
        token: str = "USDC",
    ) -> Dict[str, Any]:
        """
        Auto-detect best source chain for a wallet.
        Returns { chainId, balance, needsBridge }.
        """
        try:
            result = await self._request(
                "GET",
                "/internal/v1/agent/best-source-chain",
                params={
                    "walletId": wallet_id,
                    "amount": amount,
                    "token": token,
                },
            )
            return result.get("data", {"chainId": 42161, "balance": "0", "needsBridge": False})
        except Exception as e:
            logger.error("Failed to get best source chain", wallet_id=wallet_id, error=str(e))
            # Default to Arbitrum (no bridge)
            return {"chainId": 42161, "balance": "0", "needsBridge": False}

    async def across_bridge_to_arbitrum(
        self,
        mission_id: str,
        wallet_id: str,
        amount: str,
        source_chain_id: int,
        input_token: str = "USDC",
        recipient_address: str = "",
    ) -> Dict[str, Any]:
        """
        Bridge funds to Arbitrum via Across Protocol for mission activation.
        Called when source chain != Arbitrum.
        """
        try:
            result = await self._request(
                "POST",
                "/internal/v1/agent/across-bridge-to-arbitrum",
                json={
                    "missionId": mission_id,
                    "walletId": wallet_id,
                    "amount": amount,
                    "sourceChainId": source_chain_id,
                    "inputToken": input_token,
                    "recipientAddress": recipient_address,
                },
            )

            logger.info(
                "Across bridge initiated",
                mission_id=mission_id,
                source_chain=source_chain_id,
                bridge_op_id=result.get("data", {}).get("bridgeOperationId"),
            )

            return result.get("data", result)

        except Exception as e:
            logger.error(
                "Across bridge to Arbitrum failed",
                mission_id=mission_id,
                source_chain=source_chain_id,
                error=str(e),
            )
            return {
                "success": False,
                "error": str(e),
            }

    # ==================
    # Agent Key Signing (Fast Path — Local Vault Signing)
    # ==================

    async def sign_with_agent_key(
        self,
        mission_id: str,
        typed_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Sign EIP-712 data using the mission's per-mission agent key.

        New path: Vault ciphertext is stored in agent-service DB. We decrypt
        locally via Vault and sign in-process — no HTTP round-trip needed.

        Falls back to wallet-service for legacy missions with AES-GCM keys.

        Args:
            mission_id: Mission ID
            typed_data: Full EIP-712 structure

        Returns:
            Signature and agent address
        """
        # Try local Vault signing first (new path)
        try:
            local_result = await self._sign_locally_with_vault(mission_id, typed_data)
            if local_result is not None:
                return local_result
        except Exception as e:
            logger.debug(
                "Local Vault signing unavailable, falling back to wallet-service",
                mission_id=mission_id,
                error=str(e),
            )

        # Fall back to wallet-service (legacy missions with AES-GCM keys)
        try:
            result = await self._request(
                "POST",
                "/internal/v1/agent/sign-with-agent-key",
                json={
                    "missionId": mission_id,
                    "typedData": {
                        "domain": typed_data.get("domain"),
                        "types": typed_data.get("types"),
                        "primaryType": typed_data.get("primaryType"),
                        "message": typed_data.get("message"),
                    },
                },
            )

            logger.info(
                "Trade signed with per-mission agent key (wallet-service)",
                mission_id=mission_id,
                success=result.get("success", False),
                agent_address=result.get("agentAddress"),
            )

            return {
                "success": result.get("success", False),
                "signature": result.get("signature"),
                "agentAddress": result.get("agentAddress"),
                "typedData": typed_data,
                "nonce": typed_data.get("message", {}).get("nonce"),
                "error": result.get("error"),
            }

        except Exception as e:
            logger.error(
                "Agent key signing failed",
                mission_id=mission_id,
                error=str(e),
            )
            return {
                "success": False,
                "error": str(e),
            }

    async def _sign_locally_with_vault(
        self,
        mission_id: str,
        typed_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Sign EIP-712 data using a Vault-decrypted agent key (local, zero-latency).

        Returns None if the mission doesn't have a Vault ciphertext
        (indicates a legacy mission that should use the wallet-service path).
        """
        from app.services.database import get_agent_key_ciphertext, get_mission_by_id
        from app.config import get_settings

        settings = get_settings()
        if not settings.vault_configured:
            return None

        agent_key_enc = await get_agent_key_ciphertext(mission_id)
        if not agent_key_enc or not agent_key_enc.startswith("vault:"):
            return None  # Not a Vault ciphertext — use legacy path

        from app.services.vault.client import VaultEncryptionService
        from eth_account import Account
        from eth_account.messages import encode_typed_data

        vault = VaultEncryptionService(
            vault_url=settings.hcp_vault_url,
            vault_token=settings.hcp_vault_token,
            namespace=settings.hcp_vault_namespace,
            mount_point=settings.vault_transit_mount,
            role_id=settings.vault_role_id,
            secret_id=settings.vault_secret_id,
        )

        raw_key = await vault.decrypt_private_key(agent_key_enc)
        try:
            account = Account.from_key(raw_key)

            # Remove EIP712Domain from types if present
            types = {k: v for k, v in typed_data["types"].items() if k != "EIP712Domain"}
            primary_type = typed_data.get("primaryType", next(iter(types)))

            signable = encode_typed_data(
                primaryType=primary_type,
                domain_data=typed_data["domain"],
                types=types,
                message=typed_data["message"],
            )

            signed = account.sign_message(signable)
            signature = "0x" + signed.signature.hex()

            logger.info(
                "Trade signed with local Vault agent key",
                mission_id=mission_id,
                agent_address=account.address,
            )

            return {
                "success": True,
                "signature": signature,
                "agentAddress": account.address,
                "typedData": typed_data,
                "nonce": typed_data.get("message", {}).get("nonce"),
                "error": None,
            }
        finally:
            del raw_key
            try:
                del account
            except NameError:
                pass

    async def batch_sign_with_agent_key(
        self,
        orders: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Batch sign orders using per-mission agent keys (fast path).

        Each order's mission has its own encrypted agent key.
        All signing is local - no Turnkey API calls.

        Args:
            orders: List of {mission_id, typed_data} dicts

        Returns:
            List of signing results
        """
        if not orders:
            return []

        try:
            result = await self._request(
                "POST",
                "/internal/v1/agent/batch-sign-with-agent-key",
                json={
                    "orders": [
                        {
                            "missionId": order["mission_id"],
                            "typedData": {
                                "domain": order["typed_data"].get("domain"),
                                "types": order["typed_data"].get("types"),
                                "primaryType": order["typed_data"].get("primaryType"),
                                "message": order["typed_data"].get("message"),
                            },
                        }
                        for order in orders
                    ]
                },
            )

            results = result.get("results", [])
            success_count = sum(1 for r in results if r.get("success"))

            logger.info(
                "Batch agent key signing completed",
                total=len(orders),
                successful=success_count,
                failed=len(orders) - success_count,
            )

            return [
                {
                    "success": r.get("success", False),
                    "signature": r.get("signature"),
                    "agentAddress": r.get("agentAddress"),
                    "nonce": r.get("nonce"),
                    "error": r.get("error"),
                }
                for r in results
            ]

        except Exception as e:
            logger.error(
                "Batch agent key signing failed",
                orders_count=len(orders),
                error=str(e),
            )
            return [
                {"success": False, "error": str(e)}
                for _ in orders
            ]

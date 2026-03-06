"""
Agent Wallet Manager
Manages Hyperliquid agent approvals and trading permissions
"""

from typing import Any, Dict, List, Optional

import structlog

from .client import HyperliquidClient

logger = structlog.get_logger(__name__)


class AgentWalletManager:
    """
    Manages the agent wallet approval flow for Hyperliquid trading.

    Flow:
    1. User approves MoleApp's agent address on Hyperliquid
    2. Agent can then trade on behalf of user (no new approvals needed per trade)
    3. User can revoke at any time
    """

    def __init__(self, hyperliquid_client: Optional[HyperliquidClient] = None):
        """Initialize the agent wallet manager."""
        from app.config import get_settings

        self.settings = get_settings()
        self.hl_client = hyperliquid_client or HyperliquidClient()

        # MoleApp's agent address (the backend wallet that executes trades)
        self.agent_address = self.settings.moleapp_agent_address

        logger.info(
            "AgentWalletManager initialized",
            agent_address=self.agent_address,
        )

    async def check_user_approval(self, user_address: str) -> Dict[str, Any]:
        """
        Check if a user has approved the MoleApp agent.

        Args:
            user_address: User's wallet address

        Returns:
            Approval status and details
        """
        if not self.agent_address:
            return {
                "approved": False,
                "error": "Agent address not configured",
            }

        is_approved = await self.hl_client.check_agent_approval(
            user_address=user_address,
            agent_address=self.agent_address,
        )

        return {
            "approved": is_approved,
            "agent_address": self.agent_address,
            "user_address": user_address,
        }

    def get_approval_payload(self) -> Dict[str, Any]:
        """
        Get the payload that needs to be signed by the user to approve the agent.

        This payload should be:
        1. Sent to wallet-service
        2. Signed by user's Turnkey wallet
        3. Submitted to Hyperliquid
        """
        return self.hl_client.build_agent_approval_payload(
            agent_address=self.agent_address,
            name="MoleApp Trading Agent",
        )

    async def submit_approval(self, signed_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit a signed agent approval to Hyperliquid.

        Args:
            signed_payload: The approval payload signed by user's wallet

        Returns:
            Submission result
        """
        try:
            result = await self.hl_client._exchange_request(
                action=signed_payload["action"],
                signature=signed_payload["signature"],
                nonce=signed_payload["nonce"],
            )

            success = result.get("status") == "ok"

            logger.info(
                "Agent approval submitted",
                success=success,
                response=result,
            )

            return {
                "success": success,
                "tx_hash": result.get("response", {}).get("data", {}).get("hash"),
                "response": result,
            }

        except Exception as e:
            logger.error("Agent approval failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
            }

    async def get_trading_permissions(self, user_address: str) -> Dict[str, Any]:
        """
        Get the trading permissions for a user under agent control.

        Returns:
            Permissions and constraints
        """
        approval_status = await self.check_user_approval(user_address)

        if not approval_status["approved"]:
            return {
                "can_trade": False,
                "reason": "Agent not approved",
            }

        # Get user's account state
        account = await self.hl_client.get_account_value(user_address)
        positions = await self.hl_client.get_positions(user_address)

        return {
            "can_trade": True,
            "agent_address": self.agent_address,
            "account_value": account.get("account_value", 0),
            "withdrawable": account.get("withdrawable", 0),
            "open_positions": len(positions),
            "margin_used": account.get("total_margin_used", 0),
        }

    async def validate_trade_request(
        self,
        user_address: str,
        asset: str,
        size: float,
        leverage: int,
    ) -> Dict[str, Any]:
        """
        Validate a trade request before execution.

        Checks:
        - Agent is approved
        - User has sufficient margin
        - Position size is within limits

        Returns:
            Validation result
        """
        # Check approval
        approval = await self.check_user_approval(user_address)
        if not approval["approved"]:
            return {
                "valid": False,
                "reason": "Agent not approved for this user",
            }

        # Get account state
        account = await self.hl_client.get_account_value(user_address)
        account_value = account.get("account_value", 0)

        # Get market data for margin calculation
        market_data = await self.hl_client.get_market_data(asset)
        price = market_data.get("price", 0)
        max_leverage = market_data.get("max_leverage", 50)

        if leverage > max_leverage:
            return {
                "valid": False,
                "reason": f"Leverage {leverage}x exceeds maximum {max_leverage}x",
            }

        # Calculate required margin
        notional_value = size * price
        required_margin = notional_value / leverage

        available_margin = account.get("withdrawable", 0)

        if required_margin > available_margin:
            return {
                "valid": False,
                "reason": f"Insufficient margin. Required: ${required_margin:.2f}, Available: ${available_margin:.2f}",
            }

        return {
            "valid": True,
            "required_margin": required_margin,
            "available_margin": available_margin,
            "notional_value": notional_value,
        }

    async def get_all_approved_users(self) -> List[str]:
        """
        Get all users who have approved the MoleApp agent.

        Note: This would typically be fetched from the database,
        as Hyperliquid doesn't provide a reverse lookup.
        """
        # This should query the database for users with hyperliquidApproved = true
        # For now, return empty list
        logger.warning("get_all_approved_users not implemented - use database query")
        return []

"""
API Dependencies
Authentication and authorization dependencies for API routes
"""

import hmac
from typing import Optional

import httpx
import structlog
from fastapi import Depends, Header, HTTPException, status
from pydantic import BaseModel

logger = structlog.get_logger(__name__)


class UserInfo(BaseModel):
    """Authenticated user information."""

    user_id: str
    phone_number: Optional[str] = None
    is_verified: bool = False


class InternalServiceAuth:
    """Internal service authentication via API key."""

    def __init__(self):
        from app.config import get_settings
        self.settings = get_settings()

    async def __call__(
        self,
        x_internal_api_key: Optional[str] = Header(None, alias="X-Internal-Api-Key"),
    ) -> bool:
        """Verify internal API key for service-to-service calls."""
        if not x_internal_api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing internal API key",
            )

        expected_key = self.settings.wallet_service_api_key
        if not hmac.compare_digest(x_internal_api_key, expected_key):
            logger.warning("Invalid internal API key")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid internal API key",
            )

        return True


class KeycloakAuth:
    """Keycloak JWT authentication for user requests."""

    def __init__(self):
        from app.config import get_settings
        self.settings = get_settings()
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def __call__(
        self,
        authorization: Optional[str] = Header(None),
    ) -> UserInfo:
        """Validate Keycloak token and extract user info."""
        if not authorization:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing authorization header",
            )

        if not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization header format",
            )

        token = authorization[7:]  # Remove "Bearer " prefix

        try:
            # Validate token with user-service
            client = await self._get_client()
            response = await client.get(
                f"{self.settings.user_service_url}/api/v1/auth/validate",
                headers={"Authorization": f"Bearer {token}"},
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired token",
                )

            data = response.json()
            return UserInfo(
                user_id=data.get("userId") or data.get("sub"),
                phone_number=data.get("phoneNumber"),
                is_verified=data.get("isVerified", False),
            )

        except httpx.RequestError as e:
            logger.error("Token validation failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication service unavailable",
            )


# Dependency instances
get_internal_auth = InternalServiceAuth()
get_user_auth = KeycloakAuth()


async def get_current_user(
    user_info: UserInfo = Depends(get_user_auth),
) -> UserInfo:
    """Get the current authenticated user."""
    return user_info


async def require_verified_user(
    user_info: UserInfo = Depends(get_current_user),
) -> UserInfo:
    """Require a verified user."""
    if not user_info.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User must be verified to access this resource",
        )
    return user_info

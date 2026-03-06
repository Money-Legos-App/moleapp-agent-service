"""
Keycloak Authentication Service
Handles Client Credentials flow for service-to-service authentication
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional

import httpx
import structlog

logger = structlog.get_logger(__name__)


class KeycloakAuth:
    """
    Keycloak authentication client for service-to-service communication.

    Uses the Client Credentials flow to obtain access tokens for
    authenticating with other services (e.g., wallet-service).
    """

    def __init__(
        self,
        keycloak_url: str,
        realm: str,
        client_id: str,
        client_secret: str,
    ):
        """
        Initialize Keycloak auth client.

        Args:
            keycloak_url: Base URL of Keycloak server
            realm: Keycloak realm name
            client_id: Client ID for this service
            client_secret: Client secret for this service
        """
        self.keycloak_url = keycloak_url.rstrip("/")
        self.realm = realm
        self.client_id = client_id
        self.client_secret = client_secret

        self._token: Optional[str] = None
        self._token_expires_at: Optional[datetime] = None
        self._token_lock = asyncio.Lock()
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def token_endpoint(self) -> str:
        """Get the token endpoint URL."""
        return f"{self.keycloak_url}/realms/{self.realm}/protocol/openid-connect/token"

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def get_service_token(self) -> str:
        """
        Get a valid service access token using Client Credentials flow.

        Tokens are cached and automatically refreshed before expiry.

        Returns:
            Valid access token string

        Raises:
            httpx.HTTPError: If token request fails
        """
        async with self._token_lock:
            # Return cached token if still valid (with 30s buffer)
            if self._token and self._token_expires_at:
                if datetime.utcnow() < self._token_expires_at - timedelta(seconds=30):
                    return self._token

            # Request new token
            logger.debug("Requesting new service token from Keycloak")

            client = await self._get_client()

            try:
                response = await client.post(
                    self.token_endpoint,
                    data={
                        "grant_type": "client_credentials",
                        "client_id": self.client_id,
                        "client_secret": self.client_secret,
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                response.raise_for_status()

                token_data = response.json()
                self._token = token_data["access_token"]
                expires_in = token_data.get("expires_in", 300)
                self._token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in)

                logger.info(
                    "Service token obtained",
                    expires_in=expires_in,
                    client_id=self.client_id,
                )

                return self._token

            except httpx.HTTPStatusError as e:
                logger.error(
                    "Failed to obtain service token",
                    status_code=e.response.status_code,
                    response=e.response.text,
                )
                raise
            except Exception as e:
                logger.error(
                    "Error obtaining service token",
                    error=str(e),
                )
                raise

    async def get_auth_headers(self) -> dict:
        """
        Get authorization headers for service-to-service requests.

        Returns:
            Dict with Authorization header
        """
        token = await self.get_service_token()
        return {"Authorization": f"Bearer {token}"}


# Singleton instance
_keycloak_auth: Optional[KeycloakAuth] = None


def get_keycloak_auth() -> KeycloakAuth:
    """Get or create the Keycloak auth singleton."""
    global _keycloak_auth

    if _keycloak_auth is None:
        from app.config import get_settings
        settings = get_settings()

        _keycloak_auth = KeycloakAuth(
            keycloak_url=settings.keycloak_url,
            realm=settings.keycloak_realm,
            client_id=settings.keycloak_client_id,
            client_secret=settings.keycloak_client_secret,
        )

    return _keycloak_auth

"""
HCP Vault Transit — Encryption-as-a-Service

Encrypts/decrypts per-mission Master EOA private keys using
HCP Vault Transit Engine with AES-256-GCM96.

The raw private key is generated locally (eth_account.Account.create()),
encrypted via Vault, and stored in the database as a ciphertext string.
When signing is needed, the ciphertext is decrypted just-in-time,
used with the hyperliquid-python-sdk, then immediately dropped.

Supports two auth methods:
  1. AppRole (preferred): role_id + secret_id — auto-renews tokens
  2. Token (legacy fallback): static HCP_VAULT_TOKEN

Usage:
    vault = VaultEncryptionService(url, namespace=namespace, role_id=role_id, secret_id=secret_id)
    await vault.ensure_encryption_key()
    ciphertext = await vault.encrypt_private_key("0xabc123...")
    raw_key    = await vault.decrypt_private_key(ciphertext)
"""

import asyncio
import base64
import time
from functools import partial
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)

# One shared AES-256-GCM key for all mission Master EOA keys.
# Vault Transit handles key versioning and rotation internally.
DEFAULT_ENCRYPTION_KEY = "moleapp-master-eoa-keys"

# Renew the token when 25% of TTL remains
_RENEWAL_BUFFER_RATIO = 0.25


class VaultEncryptionService:
    """Encrypts/decrypts Master EOA private keys via HCP Vault Transit (aes256-gcm96)."""

    def __init__(
        self,
        vault_url: str,
        vault_token: str = "",
        namespace: str = "admin",
        mount_point: str = "transit",
        key_name: str = DEFAULT_ENCRYPTION_KEY,
        role_id: str = "",
        secret_id: str = "",
    ):
        import hvac

        self._hvac = hvac
        self._vault_url = vault_url
        self._namespace = namespace
        self.client = hvac.Client(
            url=vault_url,
            namespace=namespace,
        )
        self.mount = mount_point
        self.key_name = key_name
        self._role_id = role_id
        self._secret_id = secret_id
        self._auth_method = "token"
        self._token_expires_at: float = 0  # epoch seconds
        self._token_ttl: int = 0

        # AppRole auth takes priority over static token
        if role_id and secret_id:
            self._login_approle()
        elif vault_token:
            self.client.token = vault_token
            self._auth_method = "token"
        else:
            raise RuntimeError(
                "Vault auth requires either (VAULT_ROLE_ID + VAULT_SECRET_ID) or HCP_VAULT_TOKEN"
            )

        if not self.client.is_authenticated():
            raise RuntimeError("HCP Vault authentication failed — check credentials")

        logger.info(
            "Vault Encryption service initialized",
            url=vault_url,
            namespace=namespace,
            mount=mount_point,
            key_name=key_name,
            auth_method=self._auth_method,
        )

    def _login_approle(self) -> None:
        """Authenticate (or re-authenticate) via AppRole and track token expiry."""
        resp = self.client.auth.approle.login(
            role_id=self._role_id,
            secret_id=self._secret_id,
        )
        auth_data = resp.get("auth", {})
        self._token_ttl = auth_data.get("lease_duration", 3600)
        self._token_expires_at = time.monotonic() + self._token_ttl
        self._auth_method = "approle"

        logger.info(
            "Vault authenticated via AppRole",
            url=self._vault_url,
            namespace=self._namespace,
            token_ttl=self._token_ttl,
            expires_in_s=self._token_ttl,
        )

    def _ensure_token_valid(self) -> None:
        """Re-login via AppRole if the token is near expiry. No-op for static tokens."""
        if self._auth_method != "approle":
            return

        remaining = self._token_expires_at - time.monotonic()
        threshold = self._token_ttl * _RENEWAL_BUFFER_RATIO

        if remaining <= threshold:
            logger.info(
                "Vault token near expiry, re-authenticating",
                remaining_s=round(remaining, 1),
                threshold_s=round(threshold, 1),
            )
            self._login_approle()

    async def _run_sync(self, func, *args, **kwargs):
        """Run a synchronous hvac call in a thread executor to avoid blocking the event loop."""
        self._ensure_token_valid()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(func, *args, **kwargs))

    async def ensure_encryption_key(self) -> None:
        """
        Create the shared AES-256-GCM encryption key if it doesn't already exist.

        Safe to call multiple times — Vault ignores duplicate create_key calls
        for existing keys.
        """
        try:
            await self._run_sync(
                self.client.secrets.transit.create_key,
                name=self.key_name,
                key_type="aes256-gcm96",
                mount_point=self.mount,
            )
            logger.info("Vault encryption key ensured", key_name=self.key_name)
        except Exception as e:
            # Key may already exist — that's fine
            if "already exists" in str(e).lower():
                logger.debug("Vault encryption key already exists", key_name=self.key_name)
            else:
                raise

    async def encrypt_private_key(self, private_key_hex: str) -> str:
        """
        Encrypt a raw Master EOA private key.

        Args:
            private_key_hex: Hex-encoded private key (with or without 0x prefix)

        Returns:
            Vault ciphertext string (e.g., "vault:v1:AbCdEf...")
            This single string encapsulates IV, tag, and encrypted data.
        """
        clean_hex = private_key_hex.removeprefix("0x")
        raw_bytes = bytes.fromhex(clean_hex)
        b64_plaintext = base64.b64encode(raw_bytes).decode("ascii")

        result = await self._run_sync(
            self.client.secrets.transit.encrypt_data,
            name=self.key_name,
            plaintext=b64_plaintext,
            mount_point=self.mount,
        )

        ciphertext = result["data"]["ciphertext"]

        logger.debug(
            "Private key encrypted via Vault",
            ciphertext_prefix=ciphertext[:20] + "...",
        )

        return ciphertext

    async def decrypt_private_key(self, ciphertext: str) -> str:
        """
        Decrypt a Vault-encrypted Master EOA private key.

        Args:
            ciphertext: Vault ciphertext (e.g., "vault:v1:AbCdEf...")

        Returns:
            Hex-encoded private key with 0x prefix (e.g., "0xabc123...")

        IMPORTANT: Caller MUST delete the returned key from memory
        after use (del raw_key in a finally block).
        """
        result = await self._run_sync(
            self.client.secrets.transit.decrypt_data,
            name=self.key_name,
            ciphertext=ciphertext,
            mount_point=self.mount,
        )

        b64_plaintext = result["data"]["plaintext"]
        raw_bytes = base64.b64decode(b64_plaintext)
        try:
            private_key_hex = "0x" + raw_bytes.hex()
            return private_key_hex
        finally:
            del b64_plaintext, raw_bytes

    def is_healthy(self) -> bool:
        """Check if Vault is reachable and authenticated."""
        try:
            return self.client.is_authenticated()
        except Exception:
            return False

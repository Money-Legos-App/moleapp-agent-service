"""
HCP Vault Transit Engine integration — Encryption-as-a-Service.

Architecture:
- Master EOA keys are generated locally (eth_account.Account.create())
- Vault Transit encrypts the raw private key (aes256-gcm96)
- Ciphertext stored in DB; decrypted JIT for signing operations
- One shared encryption key for all missions (Vault manages rotation)
"""

from app.services.vault.client import VaultEncryptionService

__all__ = ["VaultEncryptionService"]

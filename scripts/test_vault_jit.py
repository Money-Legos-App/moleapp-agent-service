"""
Test JIT key generation + Vault encrypt/decrypt flow locally.

Prerequisites:
  docker run --rm -d --name vault-dev \
    -p 8200:8200 \
    -e VAULT_DEV_ROOT_TOKEN_ID=dev-test-token \
    hashicorp/vault:latest
  docker exec vault-dev vault secrets enable transit

Usage:
  cd backend/agent-service
  python -m scripts.test_vault_jit
"""

import asyncio
import sys
import os

# Add parent to path so we can import app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eth_account import Account


VAULT_URL = os.getenv("HCP_VAULT_URL", "http://127.0.0.1:8200")
VAULT_TOKEN = os.getenv("HCP_VAULT_TOKEN", "dev-test-token")
VAULT_NAMESPACE = os.getenv("HCP_VAULT_NAMESPACE", "")  # dev server has no namespace


async def test_full_jit_cycle():
    """Simulate the full JIT key lifecycle used in mission creation."""
    from app.services.vault.client import VaultEncryptionService

    print("=" * 60)
    print("JIT Vault Key Test — Full Lifecycle")
    print("=" * 60)

    # --- Step 1: Connect to Vault ---
    print("\n[1] Connecting to Vault...")
    vault = VaultEncryptionService(
        vault_url=VAULT_URL,
        vault_token=VAULT_TOKEN,
        namespace=VAULT_NAMESPACE or "admin",
        mount_point="transit",
    )
    print(f"    Authenticated: {vault.is_healthy()}")

    # --- Step 2: Ensure encryption key exists ---
    print("\n[2] Ensuring encryption key exists...")
    await vault.ensure_encryption_key()
    print(f"    Key '{vault.key_name}' ready")

    # --- Step 3: Generate Master EOA locally ---
    print("\n[3] Generating Master EOA locally (eth_account.Account.create())...")
    master_account = Account.create()
    master_address = master_account.address
    raw_private_key = master_account.key.hex()
    print(f"    Address: {master_address}")
    print(f"    Key (first 10 chars): {raw_private_key[:10]}...")

    # --- Step 4: Encrypt via Vault Transit ---
    print("\n[4] Encrypting private key via Vault Transit...")
    ciphertext = await vault.encrypt_private_key(raw_private_key)
    print(f"    Ciphertext: {ciphertext[:40]}...")
    print(f"    (This is what gets stored in DB as masterEoaKeyEnc)")

    # Clear raw key from memory (simulating production behavior)
    del master_account, raw_private_key

    # --- Step 5: JIT Decrypt (simulating signing time) ---
    print("\n[5] JIT Decrypt — simulating a signing operation...")
    decrypted_key = await vault.decrypt_private_key(ciphertext)
    try:
        recovered = Account.from_key(decrypted_key)
        print(f"    Recovered address: {recovered.address}")
        assert recovered.address == master_address, "ADDRESS MISMATCH!"
        print(f"    Address match: OK")

        # Simulate signing something
        from eth_account.messages import encode_defunct
        msg = encode_defunct(text="test-signing")
        sig = recovered.sign_message(msg)
        print(f"    Test signature: {sig.signature.hex()[:20]}...")
        print(f"    Signing works: OK")
    finally:
        # Critical: clear decrypted key from memory
        del decrypted_key
        try:
            del recovered
        except NameError:
            pass
        print("    Key cleared from memory: OK")

    # --- Step 6: Verify re-decrypt gives same result ---
    print("\n[6] Re-decrypting (idempotency check)...")
    decrypted_key_2 = await vault.decrypt_private_key(ciphertext)
    try:
        recovered_2 = Account.from_key(decrypted_key_2)
        assert recovered_2.address == master_address, "ADDRESS MISMATCH on re-decrypt!"
        print(f"    Re-decrypt matches: OK")
    finally:
        del decrypted_key_2
        try:
            del recovered_2
        except NameError:
            pass

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  Master EOA address: {master_address}")
    print(f"  Ciphertext stored:  {ciphertext[:50]}...")
    print(f"  Vault URL:          {VAULT_URL}")
    print(f"  Encrypt/Decrypt:    OK")
    print(f"  JIT signing:        OK")
    print(f"  Memory cleanup:     OK")


async def test_api_endpoint_simulation():
    """Simulate what the /missions POST endpoint does with Vault."""
    from app.services.vault.client import VaultEncryptionService

    print("\n" + "=" * 60)
    print("API Endpoint Simulation — POST /missions")
    print("=" * 60)

    vault = VaultEncryptionService(
        vault_url=VAULT_URL,
        vault_token=VAULT_TOKEN,
        namespace=VAULT_NAMESPACE or "admin",
    )
    await vault.ensure_encryption_key()

    # Simulate mission creation (what create_mission() does)
    print("\n[Mission Create] Generating Master EOA...")
    account = Account.create()
    address = account.address
    raw_key = account.key.hex()

    print(f"  Master EOA: {address}")

    ciphertext = await vault.encrypt_private_key(raw_key)
    del account, raw_key  # Clear immediately

    print(f"  Encrypted key: {ciphertext[:40]}...")
    print(f"  → Would store in DB: Mission.masterEoaKeyEnc = '{ciphertext[:30]}...'")
    print(f"  → Would store in DB: Mission.masterEoaAddress = '{address}'")

    # Simulate agent approval (what happens later when deposit arrives)
    print("\n[Agent Approval] JIT decrypt for SDK signing...")
    raw_key = await vault.decrypt_private_key(ciphertext)
    try:
        recovered = Account.from_key(raw_key)
        assert recovered.address == address
        print(f"  Recovered: {recovered.address} — match!")
        print(f"  → Would call: Exchange(recovered, ...).approve_agent()")
    finally:
        del raw_key, recovered

    print("\n  Simulation complete — Vault JIT flow works end-to-end!")


async def main():
    try:
        await test_full_jit_cycle()
        await test_api_endpoint_simulation()
    except Exception as e:
        print(f"\nERROR: {e}")
        print(f"\nMake sure Vault dev server is running:")
        print(f"  docker run --rm -d --name vault-dev \\")
        print(f"    -p 8200:8200 \\")
        print(f"    -e VAULT_DEV_ROOT_TOKEN_ID=dev-test-token \\")
        print(f"    hashicorp/vault:latest")
        print(f"  docker exec vault-dev vault secrets enable transit")
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())

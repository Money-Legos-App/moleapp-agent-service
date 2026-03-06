"""
Deposit Polling Task

Monitors missions in DEPOSITING status and transitions them to ACTIVE.

Flow (Vault-only):
  DEPOSITING → (funds land on HL L1) → approve_agent via Vault → ACTIVE
"""

from typing import Dict, Any, List

import structlog

logger = structlog.get_logger(__name__)

# Minimum seconds a mission must stay in DEPOSITING before we approve,
# to prevent front-running between balance check and agent approval.
_MIN_DEPOSIT_DWELL_SECONDS = 30


async def poll_pending_deposits() -> Dict[str, int]:
    """
    Poll missions in DEPOSITING status to check if funds have landed on HL L1.

    Flow:
    1. Fetch all missions with status=DEPOSITING
    2. For each, check the Master EOA's HL L1 balance
    3. If balance >= depositAmount, approve agent via Vault and activate

    Returns:
        Dictionary with counts of transitions
    """
    from app.services.hyperliquid import HyperliquidClient
    from app.services.wallet import TurnkeyBridge

    logger.debug("Polling pending deposits")

    hl_client = HyperliquidClient()
    wallet_bridge = TurnkeyBridge()

    result = {
        "deposits_confirmed": 0,
        "activated": 0,
        "errors": 0,
    }

    try:
        depositing_missions = await _get_depositing_missions()

        if not depositing_missions:
            return result

        logger.info("Checking depositing missions", count=len(depositing_missions))

        for mission in depositing_missions:
            mission_id = mission["id"]
            master_eoa = mission.get("master_eoa_address")

            if not master_eoa:
                logger.warning("No master EOA address for depositing mission", mission_id=mission_id)
                continue

            try:
                # Check HL L1 balance on the Master EOA
                account = await hl_client.get_account_value(master_eoa)
                hl_balance = account.get("account_value", 0)
                expected_deposit = float(mission.get("initial_capital", 0))

                logger.debug(
                    "Checking HL balance",
                    mission_id=mission_id,
                    address=master_eoa,
                    hl_balance=hl_balance,
                    expected=expected_deposit,
                )

                if hl_balance >= expected_deposit * 0.95:  # 95% threshold for rounding
                    # Verify deposit dwell time to prevent race between
                    # balance check and agent approval
                    depositing_since = mission.get("updated_at")
                    if depositing_since:
                        import datetime
                        now = datetime.datetime.utcnow()
                        if hasattr(depositing_since, "timestamp"):
                            elapsed = (now - depositing_since).total_seconds()
                        else:
                            elapsed = _MIN_DEPOSIT_DWELL_SECONDS  # assume safe if no timestamp
                        if elapsed < _MIN_DEPOSIT_DWELL_SECONDS:
                            logger.debug(
                                "Deposit too recent, deferring activation",
                                mission_id=mission_id,
                                elapsed_s=elapsed,
                            )
                            continue

                    # Double-check: verify HL balance a second time right before approval
                    # to narrow the TOCTOU window
                    recheck = await hl_client.get_account_value(master_eoa)
                    recheck_balance = recheck.get("account_value", 0)
                    if recheck_balance < expected_deposit * 0.95:
                        logger.warning(
                            "HL balance dropped between checks, skipping activation",
                            mission_id=mission_id,
                            first_check=hl_balance,
                            second_check=recheck_balance,
                        )
                        continue

                    result["deposits_confirmed"] += 1
                    logger.info(
                        "Deposit confirmed on HL L1 (double-checked)",
                        mission_id=mission_id,
                        balance=recheck_balance,
                    )

                    # Approve agent via Vault SDK and activate
                    activated = await _approve_and_activate_via_vault(
                        mission_id=mission_id,
                        encrypted_master_key=mission.get("master_eoa_key_enc", ""),
                        wallet_bridge=wallet_bridge,
                        hl_client=hl_client,
                    )

                    if activated:
                        result["activated"] += 1

            except Exception as e:
                logger.error(
                    "Error polling deposit for mission",
                    mission_id=mission_id,
                    error=str(e),
                )
                result["errors"] += 1
                continue

    finally:
        await hl_client.close()
        await wallet_bridge.close()

    if any(result.values()):
        logger.info("Deposit polling completed", **result)

    return result


async def _approve_and_activate_via_vault(
    mission_id: str,
    encrypted_master_key: str,
    wallet_bridge,
    hl_client,
) -> bool:
    """
    Approve the agent on Hyperliquid using the SDK with a JIT-decrypted
    Master EOA key, then transition directly to ACTIVE.

    Requires Vault Transit to be configured. Fails hard if not.

    Returns True if activation succeeded.
    """
    from app.services.vault.client import VaultEncryptionService
    from app.services.hyperliquid.agent_approval import approve_agent_with_sdk
    from app.services.database import claim_mission_for_activation
    from app.config import get_settings

    settings = get_settings()

    # Atomically claim this mission — prevents race between polling and manual activation
    claimed = await claim_mission_for_activation(mission_id)
    if not claimed:
        logger.info("Mission already claimed for activation by another process", mission_id=mission_id)
        return False

    # Vault is mandatory — fail hard if not configured
    if not settings.vault_configured:
        logger.error(
            "Vault not configured — cannot approve agent. Set HCP_VAULT_URL and credentials.",
            mission_id=mission_id,
        )
        await wallet_bridge.update_mission_status(
            mission_id=mission_id,
            status="PENDING",
            metadata={"failureReason": "Vault not configured"},
        )
        return False

    if not encrypted_master_key:
        logger.error(
            "No encrypted master key for mission — cannot approve agent",
            mission_id=mission_id,
        )
        await wallet_bridge.update_mission_status(
            mission_id=mission_id,
            status="PENDING",
            metadata={"failureReason": "Missing encrypted master key"},
        )
        return False

    try:
        vault = VaultEncryptionService(
            vault_url=settings.hcp_vault_url,
            vault_token=settings.hcp_vault_token,
            namespace=settings.hcp_vault_namespace,
            mount_point=settings.vault_transit_mount,
            role_id=settings.vault_role_id,
            secret_id=settings.vault_secret_id,
        )

        # Approve agent via SDK (decrypts key JIT, generates agent key)
        approval = await approve_agent_with_sdk(
            mission_id=mission_id,
            vault=vault,
            encrypted_master_key=encrypted_master_key,
            is_mainnet=settings.hyperliquid_mainnet,
        )

        if not approval.get("success"):
            logger.error(
                "SDK agent approval failed — reverting to PENDING",
                mission_id=mission_id,
                error=approval.get("error"),
            )
            await wallet_bridge.update_mission_status(
                mission_id=mission_id,
                status="PENDING",
                metadata={"failureReason": f"SDK approval failed: {approval.get('error')}"},
            )
            return False

        # Store agent key as Vault ciphertext in DB (key never leaves agent-service)
        from app.services.database import update_mission_status
        await update_mission_status(
            mission_id=mission_id,
            new_status="ACTIVE",
            extra_fields={
                "agentPrivateKeyEnc": approval["agent_key_ciphertext"],
                "agentAddress": approval["agent_address"],
                "hyperliquidApproved": True,
            },
        )

        # Only send address to wallet-service (no key material over HTTP)
        await wallet_bridge.store_agent_address(
            mission_id=mission_id,
            agent_address=approval["agent_address"],
        )

        logger.info(
            "Mission activated via SDK approval",
            mission_id=mission_id,
            agent_address=approval["agent_address"],
        )
        return True

    except Exception as e:
        logger.error(
            "SDK approval exception — reverting to PENDING",
            mission_id=mission_id,
            error=str(e),
        )
        await wallet_bridge.update_mission_status(
            mission_id=mission_id,
            status="PENDING",
            metadata={"failureReason": f"Exception: {str(e)}"},
        )
        return False


async def _get_depositing_missions() -> List[Dict[str, Any]]:
    """Fetch missions with DEPOSITING status."""
    from app.services.database import get_missions_by_status
    try:
        return await get_missions_by_status("DEPOSITING")
    except Exception as e:
        logger.error("Failed to fetch depositing missions", error=str(e))
        return []

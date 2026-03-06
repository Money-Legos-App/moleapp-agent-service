"""
Deposit Worker (Event-Driven with Exponential Backoff)

Replaces the global deposit polling cron with per-mission jobs.
When a mission is activated, a single CheckDeposit job is enqueued.
If the deposit hasn't landed yet, the job retries with increasing delays:
30s -> 60s -> 120s -> 300s (capped), up to ~30 minutes.
"""

from datetime import timedelta

import structlog
from arq import Retry

logger = structlog.get_logger(__name__)

# Backoff schedule in seconds: 30s, 1m, 2m, 5m (then stays at 5m)
BACKOFF_SCHEDULE = [30, 60, 120, 300]


def _backoff_delay(attempt: int) -> int:
    """Get backoff delay in seconds for the given attempt number."""
    if attempt < len(BACKOFF_SCHEDULE):
        return BACKOFF_SCHEDULE[attempt]
    return BACKOFF_SCHEDULE[-1]  # Cap at max


async def check_deposit_for_mission(
    ctx: dict,
    mission_id: str,
    expected_amount: float,
    user_address: str,
    agent_address: str,
) -> dict:
    """
    arq task: Check if a mission's deposit has landed on Hyperliquid L1.

    If confirmed (balance >= 95% of expected), initiates agent approval
    and transitions mission to ACTIVE.

    If still pending, raises Retry with exponential backoff.

    Uses arq's internal job_try (from ctx) for attempt tracking instead
    of a manual attempt parameter, since arq Retry replays with the same args.
    """
    from app.services.hyperliquid import HyperliquidClient
    from app.services.wallet import TurnkeyBridge
    from app.tasks.deposit_polling import _initiate_agent_approval
    from app.config import get_settings

    settings = get_settings()

    # arq's job_try is 1-based, convert to 0-based attempt
    attempt = ctx.get("job_try", 1) - 1

    logger.info(
        "Checking deposit",
        mission_id=mission_id,
        attempt=attempt,
        expected=expected_amount,
        user_address=user_address,
    )

    # Rate limiting is handled at the client level (HyperliquidRateLimiter)
    hl_client = HyperliquidClient()
    wallet_bridge = TurnkeyBridge()

    try:
        account = await hl_client.get_account_value(user_address)
        hl_balance = account.get("account_value", 0)

        logger.info(
            "HL balance check",
            mission_id=mission_id,
            hl_balance=hl_balance,
            expected=expected_amount,
            threshold=expected_amount * 0.95,
            user_address=user_address,
        )

        if hl_balance >= expected_amount * 0.95:
            # Deposit confirmed! Initiate agent approval
            logger.info(
                "Deposit confirmed on HL L1",
                mission_id=mission_id,
                balance=hl_balance,
            )

            await _initiate_agent_approval(
                mission_id=mission_id,
                wallet_bridge=wallet_bridge,
                hl_client=hl_client,
                agent_address=agent_address,
            )

            return {
                "status": "confirmed",
                "mission_id": mission_id,
                "balance": hl_balance,
                "attempts": attempt + 1,
            }

        # Still pending — retry with backoff
        if attempt + 1 >= settings.deposit_check_max_attempts:
            logger.error(
                "Deposit check max attempts reached",
                mission_id=mission_id,
                attempts=attempt + 1,
                last_balance=hl_balance,
            )
            return {
                "status": "timeout",
                "mission_id": mission_id,
                "balance": hl_balance,
                "attempts": attempt + 1,
            }

        delay = _backoff_delay(attempt)
        logger.info(
            "Deposit not yet confirmed, scheduling retry",
            mission_id=mission_id,
            attempt=attempt,
            next_delay_seconds=delay,
        )

        raise Retry(defer=timedelta(seconds=delay))

    finally:
        await hl_client.close()
        await wallet_bridge.close()


async def enqueue_deposit_check(
    mission_id: str,
    expected_amount: float,
    user_address: str,
    agent_address: str,
    initial_delay_seconds: int = 120,
) -> None:
    """
    Enqueue the first deposit check job with an initial delay.
    Called from the activate_mission endpoint.
    """
    from app.services.execution_queue import get_arq_pool

    arq_pool = await get_arq_pool()
    await arq_pool.enqueue_job(
        "check_deposit_for_mission",
        mission_id=mission_id,
        expected_amount=expected_amount,
        user_address=user_address,
        agent_address=agent_address,
        _defer_by=timedelta(seconds=initial_delay_seconds),
    )

    logger.info(
        "Deposit check enqueued",
        mission_id=mission_id,
        initial_delay_seconds=initial_delay_seconds,
    )

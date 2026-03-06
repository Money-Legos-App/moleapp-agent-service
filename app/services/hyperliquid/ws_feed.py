"""
Hyperliquid WebSocket Market Data Feed

Subscribes to the allMids WebSocket channel for real-time price updates.
Updates the Redis cache continuously, eliminating the need for periodic
REST API calls for market prices.

Resilience:
- Auto-reconnect with exponential backoff + jitter (1s -> 60s cap)
- Fallback to REST polling if WS stays disconnected for > 30 seconds
- Heartbeat via websockets library ping (every 50s per HL SDK convention)
"""

import asyncio
import json
import random
import time
from typing import Any, Callable, Coroutine, Dict, Optional

import structlog
import websockets
from websockets.exceptions import ConnectionClosed

logger = structlog.get_logger(__name__)


class HyperliquidWSFeed:
    """Manages a WebSocket connection to Hyperliquid for real-time allMids data."""

    MAINNET_WS = "wss://api.hyperliquid.xyz/ws"
    TESTNET_WS = "wss://api.hyperliquid-testnet.xyz/ws"

    PING_INTERVAL = 50  # seconds, per HL SDK convention
    MAX_RECONNECT_DELAY = 60  # seconds
    STALE_THRESHOLD = 30  # seconds before considered unhealthy

    def __init__(
        self,
        is_mainnet: bool = False,
        on_prices_update: Optional[Callable[[Dict[str, str]], Coroutine[Any, Any, None]]] = None,
        max_reconnect_delay: int = 60,
    ):
        self._ws_url = self.MAINNET_WS if is_mainnet else self.TESTNET_WS
        self._on_prices_update = on_prices_update
        self._max_reconnect_delay = max_reconnect_delay
        self._ws = None
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._last_message_at: float = 0
        self._reconnect_attempts: int = 0
        self._latest_mids: Dict[str, str] = {}

    @property
    def is_healthy(self) -> bool:
        """True if WS is connected and received data recently."""
        if not self._running or self._ws is None:
            return False
        return (time.time() - self._last_message_at) < self.STALE_THRESHOLD

    @property
    def latest_mids(self) -> Dict[str, str]:
        """Latest mid prices from WebSocket."""
        return self._latest_mids

    async def start(self) -> None:
        """Start the WebSocket feed as a background task."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("WS feed started", url=self._ws_url)

    async def stop(self) -> None:
        """Stop the WebSocket feed."""
        self._running = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("WS feed stopped")

    async def _run_loop(self) -> None:
        """Main reconnection loop with exponential backoff + jitter."""
        while self._running:
            try:
                await self._connect_and_listen()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._ws = None
                self._reconnect_attempts += 1
                delay = min(2 ** self._reconnect_attempts, self._max_reconnect_delay)
                # Add jitter: +/- 25%
                jitter = delay * 0.25 * (2 * random.random() - 1)
                delay = max(1.0, delay + jitter)
                logger.warning(
                    "WS disconnected, reconnecting",
                    error=str(e),
                    attempt=self._reconnect_attempts,
                    delay_seconds=round(delay, 1),
                )
                await asyncio.sleep(delay)

    async def _connect_and_listen(self) -> None:
        """Connect, subscribe to allMids, and process messages."""
        async with websockets.connect(
            self._ws_url,
            ping_interval=self.PING_INTERVAL,
            ping_timeout=10,
            close_timeout=5,
        ) as ws:
            self._ws = ws
            self._reconnect_attempts = 0
            logger.info("WS connected", url=self._ws_url)

            # Subscribe to allMids
            subscribe_msg = json.dumps({
                "method": "subscribe",
                "subscription": {"type": "allMids"},
            })
            await ws.send(subscribe_msg)

            # Process messages
            async for raw_msg in ws:
                if not self._running:
                    break

                self._last_message_at = time.time()

                try:
                    msg = json.loads(raw_msg)
                    channel = msg.get("channel")

                    if channel == "allMids":
                        mids = msg.get("data", {}).get("mids", {})
                        if mids:
                            self._latest_mids = mids
                            if self._on_prices_update:
                                try:
                                    await self._on_prices_update(mids)
                                except Exception as cb_err:
                                    logger.error(
                                        "WS price update callback failed",
                                        error=str(cb_err),
                                    )

                    elif channel == "subscriptionResponse":
                        logger.info(
                            "WS subscription confirmed",
                            method=msg.get("data", {}).get("method"),
                        )

                except (json.JSONDecodeError, KeyError) as e:
                    logger.debug("WS message parse error", error=str(e))

        self._ws = None

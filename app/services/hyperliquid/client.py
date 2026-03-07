"""
Hyperliquid Exchange Client
Handles market data, order placement, and position management
"""

import asyncio
import logging
import time
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
    before_sleep_log,
)

logger = structlog.get_logger(__name__)


def _retry_if_transient(exc: BaseException) -> bool:
    """Only retry on transient errors: 429, 5xx, and network errors."""
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    if isinstance(exc, (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout,
                        httpx.PoolTimeout, httpx.ConnectTimeout)):
        return True
    return False


# ==================
# EIP-712 Type Definitions for Hyperliquid
# ==================

# Hyperliquid uses EIP-712 structured data signing
# These types must match Hyperliquid's L1 protocol exactly

HYPERLIQUID_EIP712_DOMAIN = {
    "name": "Exchange",
    "version": "1",
    "chainId": 1337,  # Hyperliquid L1 chain ID (testnet: 1337, mainnet: 1)
    "verifyingContract": "0x0000000000000000000000000000000000000000",
}

# Order action types for EIP-712
ORDER_ACTION_TYPES = {
    "EIP712Domain": [
        {"name": "name", "type": "string"},
        {"name": "version", "type": "string"},
        {"name": "chainId", "type": "uint256"},
        {"name": "verifyingContract", "type": "address"},
    ],
    "Order": [
        {"name": "asset", "type": "uint32"},
        {"name": "isBuy", "type": "bool"},
        {"name": "limitPx", "type": "uint64"},
        {"name": "sz", "type": "uint64"},
        {"name": "reduceOnly", "type": "bool"},
        {"name": "orderType", "type": "uint8"},
        {"name": "cloid", "type": "bytes16"},
    ],
    "OrderAction": [
        {"name": "orders", "type": "Order[]"},
        {"name": "grouping", "type": "uint8"},
        {"name": "nonce", "type": "uint64"},
    ],
}

# Agent approval types for EIP-712
AGENT_APPROVAL_TYPES = {
    "EIP712Domain": [
        {"name": "name", "type": "string"},
        {"name": "version", "type": "string"},
        {"name": "chainId", "type": "uint256"},
        {"name": "verifyingContract", "type": "address"},
    ],
    "Agent": [
        {"name": "source", "type": "string"},
        {"name": "connectionId", "type": "bytes32"},
    ],
    "ApproveAgent": [
        {"name": "hyperliquidChain", "type": "string"},
        {"name": "agentAddress", "type": "address"},
        {"name": "agentName", "type": "string"},
        {"name": "nonce", "type": "uint64"},
    ],
}


class HyperliquidClient:
    """
    Client for Hyperliquid DEX API.

    Hyperliquid is a perpetuals DEX on its own L1.
    - No gas fees for trading
    - Orders are signed and submitted to the sequencer
    - Agent approval allows trading on behalf of users
    """

    def __init__(
        self,
        api_url: Optional[str] = None,
        is_mainnet: Optional[bool] = None,
    ):
        """Initialize the Hyperliquid client."""
        from app.config import get_settings, HyperliquidConfig

        settings = get_settings()
        self.is_mainnet = is_mainnet if is_mainnet is not None else settings.hyperliquid_mainnet
        self.api_url = api_url or HyperliquidConfig.get_api_url(self.is_mainnet)

        self._client: Optional[httpx.AsyncClient] = None
        self._asset_index_cache: Optional[Dict[str, int]] = None

        logger.info(
            "HyperliquidClient initialized",
            api_url=self.api_url,
            is_mainnet=self.is_mainnet,
        )

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                headers={"Content-Type": "application/json"},
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    @retry(
        retry=retry_if_exception(_retry_if_transient),
        stop=stop_after_attempt(4),
        wait=wait_exponential_jitter(initial=2, max=30, jitter=5),
        before_sleep=before_sleep_log(logger, logging.INFO),
    )
    async def _info_request(self, request_type: str, payload: Dict = None) -> Any:
        """Make an info request to Hyperliquid."""
        from app.services.hyperliquid.rate_limiter import get_hl_rate_limiter, EndpointType

        await get_hl_rate_limiter().acquire(EndpointType.INFO, request_type)

        client = await self._get_client()

        data = {"type": request_type}
        if payload:
            data.update(payload)

        response = await client.post(
            f"{self.api_url}/info",
            json=data,
        )
        response.raise_for_status()
        return response.json()

    @retry(
        retry=retry_if_exception(_retry_if_transient),
        stop=stop_after_attempt(2),
        wait=wait_exponential_jitter(initial=2, max=15, jitter=3),
        before_sleep=before_sleep_log(logger, logging.INFO),
    )
    async def _exchange_request(self, action: Dict, signature: Any, nonce: int) -> Any:
        """Make an exchange request to Hyperliquid (requires signature)."""
        from app.services.hyperliquid.rate_limiter import get_hl_rate_limiter, EndpointType

        batch_length = len(action.get("orders", [action]))
        await get_hl_rate_limiter().acquire(EndpointType.EXCHANGE, "exchange", batch_length)

        client = await self._get_client()

        response = await client.post(
            f"{self.api_url}/exchange",
            json={
                "action": action,
                "nonce": nonce,
                "signature": signature,
            },
        )
        response.raise_for_status()
        return response.json()

    # ==================
    # Market Data Methods
    # ==================

    async def get_all_mids(self) -> Dict[str, float]:
        """Get mid prices for all assets."""
        data = await self._info_request("allMids")
        return {k: float(v) for k, v in data.items()}

    async def get_mark_prices(self, assets: Optional[List[str]] = None) -> Dict[str, float]:
        """Get mark prices for specified assets or all assets."""
        mids = await self.get_all_mids()

        if assets:
            return {a: mids.get(a, 0) for a in assets}
        return mids

    async def get_market_data(self, asset: str) -> Dict[str, Any]:
        """
        Get comprehensive market data for an asset.

        Uses the metaAndAssetCtxs endpoint which returns both universe metadata
        AND per-asset context (markPx, funding, openInterest, dayNtlVlm) in a
        single call — more efficient than separate meta + allMids requests.

        Returns:
            Dictionary with price, volume, spread, funding rate, OI, etc.
        """
        coin = asset.replace("-USD", "")

        # Single call: returns [meta, [assetCtx, ...]] — universe + per-asset context
        meta_and_ctxs = await self._info_request("metaAndAssetCtxs")

        meta = meta_and_ctxs[0] if len(meta_and_ctxs) > 0 else {}
        asset_ctxs = meta_and_ctxs[1] if len(meta_and_ctxs) > 1 else []
        universe = meta.get("universe", [])

        # Find asset index in universe and its context
        asset_info = None
        asset_ctx = None
        for idx, uni in enumerate(universe):
            if uni.get("name") == coin:
                asset_info = uni
                if idx < len(asset_ctxs):
                    asset_ctx = asset_ctxs[idx]
                break

        if not asset_info:
            logger.warning("Asset not found in Hyperliquid universe", asset=asset)

        # Extract data from asset context (real-time per-asset data)
        if asset_ctx:
            mark_price = float(asset_ctx.get("markPx", 0))
            funding_rate = float(asset_ctx.get("funding", 0))
            open_interest = float(asset_ctx.get("openInterest", 0))
            volume_24h = float(asset_ctx.get("dayNtlVlm", 0))
            prev_day_px = float(asset_ctx.get("prevDayPx", 0))
        else:
            mark_price = 0
            funding_rate = 0
            open_interest = 0
            volume_24h = 0
            prev_day_px = 0

        # Calculate 24h price change from prevDayPx
        price_change_24h = ((mark_price - prev_day_px) / prev_day_px * 100) if prev_day_px else 0

        # Populate asset index cache while we have the universe data
        if self._asset_index_cache is None:
            self._asset_index_cache = {
                u["name"]: i for i, u in enumerate(universe)
            }

        # Get L2 orderbook for spread (still needed — not in metaAndAssetCtxs)
        l2_data = await self._info_request("l2Book", {"coin": coin})
        bids = l2_data.get("levels", [[]])[0]
        asks = l2_data.get("levels", [[], []])[1]

        best_bid = float(bids[0]["px"]) if bids else mark_price * 0.999
        best_ask = float(asks[0]["px"]) if asks else mark_price * 1.001
        spread = (best_ask - best_bid) / mark_price * 100 if mark_price else 0

        return {
            "asset": asset,
            "price": mark_price,
            "mark_price": mark_price,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": spread,
            "volume_24h": volume_24h,
            "price_change_24h": price_change_24h,
            "funding_rate": funding_rate,
            "open_interest": open_interest,
            "max_leverage": asset_info.get("maxLeverage", 50) if asset_info else 50,
        }

    async def get_bulk_market_data(self, assets: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get market data for multiple assets efficiently.

        Fetches metaAndAssetCtxs once (1 API call), then L2 books
        concurrently (N calls). Total: 1 + N instead of 2N.
        """
        # 1) Single call for all universe + context data
        meta_and_ctxs = await self._info_request("metaAndAssetCtxs")
        meta = meta_and_ctxs[0] if len(meta_and_ctxs) > 0 else {}
        asset_ctxs = meta_and_ctxs[1] if len(meta_and_ctxs) > 1 else []
        universe = meta.get("universe", [])

        # Build coin -> index lookup
        coin_index = {u["name"]: i for i, u in enumerate(universe)}
        self._asset_index_cache = coin_index

        # Filter to only assets that exist in the HL universe
        valid_assets = []
        for asset in assets:
            coin = asset.replace("-USD", "")
            if coin in coin_index:
                valid_assets.append(asset)
            else:
                logger.warning("Asset not found in Hyperliquid universe, skipping", asset=asset)

        # 2) Fetch L2 books concurrently for valid assets only
        async def _get_l2(coin: str):
            return coin, await self._info_request("l2Book", {"coin": coin})

        coins = [a.replace("-USD", "") for a in valid_assets]
        l2_results = await asyncio.gather(
            *[_get_l2(c) for c in coins],
            return_exceptions=True,
        )
        l2_map = {}
        for result in l2_results:
            if isinstance(result, tuple):
                l2_map[result[0]] = result[1]

        # 3) Assemble per-asset data
        results: Dict[str, Dict[str, Any]] = {}
        for asset in valid_assets:
            coin = asset.replace("-USD", "")
            idx = coin_index.get(coin)

            asset_info = universe[idx] if idx is not None and idx < len(universe) else None
            asset_ctx = asset_ctxs[idx] if idx is not None and idx < len(asset_ctxs) else None

            if asset_ctx:
                mark_price = float(asset_ctx.get("markPx", 0))
                funding_rate = float(asset_ctx.get("funding", 0))
                open_interest = float(asset_ctx.get("openInterest", 0))
                volume_24h = float(asset_ctx.get("dayNtlVlm", 0))
                prev_day_px = float(asset_ctx.get("prevDayPx", 0))
            else:
                mark_price = 0
                funding_rate = 0
                open_interest = 0
                volume_24h = 0
                prev_day_px = 0

            price_change_24h = ((mark_price - prev_day_px) / prev_day_px * 100) if prev_day_px else 0

            l2_data = l2_map.get(coin, {})
            bids = l2_data.get("levels", [[]])[0]
            asks = l2_data.get("levels", [[], []])[1]
            best_bid = float(bids[0]["px"]) if bids else mark_price * 0.999
            best_ask = float(asks[0]["px"]) if asks else mark_price * 1.001
            spread = (best_ask - best_bid) / mark_price * 100 if mark_price else 0

            # Orderbook depth imbalance (first 5 levels)
            bid_depth = sum(float(b.get("sz", 0)) for b in bids[:5])
            ask_depth = sum(float(a.get("sz", 0)) for a in asks[:5])
            total_depth = bid_depth + ask_depth
            if total_depth > 0:
                bid_imbalance = round((bid_depth - ask_depth) / total_depth * 100, 1)
            else:
                bid_imbalance = 0.0

            results[asset] = {
                "asset": asset,
                "price": mark_price,
                "mark_price": mark_price,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "spread": spread,
                "volume_24h": volume_24h,
                "price_change_24h": price_change_24h,
                "funding_rate": funding_rate,
                "open_interest": open_interest,
                "bid_imbalance_pct": bid_imbalance,
                "max_leverage": asset_info.get("maxLeverage", 50) if asset_info else 50,
            }

        return results

    async def get_candle_data(
        self, coin: str, interval: str = "1h", lookback: int = 50
    ) -> List[Dict[str, float]]:
        """
        Fetch OHLCV candle data from Hyperliquid.

        Args:
            coin: Asset symbol without suffix (e.g., "ETH")
            interval: Candle interval ("15m", "1h", "4h", "1d")
            lookback: Number of candles to fetch

        Returns:
            List of dicts: [{open, high, low, close, volume, time}, ...]
        """
        import time as _time

        now_ms = int(_time.time() * 1000)
        # Map interval string to milliseconds for startTime calculation
        interval_ms = {
            "15m": 15 * 60 * 1000,
            "1h": 3600 * 1000,
            "4h": 4 * 3600 * 1000,
            "1d": 24 * 3600 * 1000,
        }
        ms = interval_ms.get(interval, 3600 * 1000)
        start_ms = now_ms - (lookback * ms)

        data = await self._info_request("candleSnapshot", {
            "coin": coin,
            "interval": interval,
            "startTime": start_ms,
            "endTime": now_ms,
        })

        candles = []
        for c in data:
            candles.append({
                "open": float(c.get("o", 0)),
                "high": float(c.get("h", 0)),
                "low": float(c.get("l", 0)),
                "close": float(c.get("c", 0)),
                "volume": float(c.get("v", 0)),
                "time": c.get("t", 0),
            })
        return candles

    @staticmethod
    def compute_technical_summary(candles: List[Dict[str, float]], interval: str) -> Dict[str, Any]:
        """
        Compute key technical indicators from candle data.
        All math is done in Python — the LLM receives pre-computed summaries.

        Returns:
            Dict with RSI-14, EMA-20/50 trend, ATR-14, and volume profile.
        """
        if len(candles) < 20:
            return {"interval": interval, "available": False}

        closes = [c["close"] for c in candles]
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        volumes = [c["volume"] for c in candles]

        # RSI-14
        rsi = 50.0
        if len(closes) >= 15:
            gains, losses = [], []
            for i in range(1, min(15, len(closes))):
                delta = closes[-i] - closes[-i - 1]
                gains.append(max(delta, 0))
                losses.append(max(-delta, 0))
            avg_gain = sum(gains) / len(gains) if gains else 0
            avg_loss = sum(losses) / len(losses) if losses else 0
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            elif avg_gain > 0:
                rsi = 100.0

        # EMA helper
        def _ema(data, period):
            if len(data) < period:
                return sum(data) / len(data) if data else 0
            k = 2 / (period + 1)
            ema_val = sum(data[:period]) / period
            for price in data[period:]:
                ema_val = price * k + ema_val * (1 - k)
            return ema_val

        ema_20 = _ema(closes, 20)
        ema_50 = _ema(closes, min(50, len(closes)))
        current_price = closes[-1]

        if ema_20 > ema_50 * 1.002:
            trend = "bullish"
        elif ema_20 < ema_50 * 0.998:
            trend = "bearish"
        else:
            trend = "neutral"

        # ATR-14
        true_ranges = []
        for i in range(1, min(15, len(closes))):
            tr = max(
                highs[-i] - lows[-i],
                abs(highs[-i] - closes[-i - 1]),
                abs(lows[-i] - closes[-i - 1]),
            )
            true_ranges.append(tr)
        atr = sum(true_ranges) / len(true_ranges) if true_ranges else 0
        atr_pct = (atr / current_price * 100) if current_price > 0 else 0

        # Volume profile: current vs average
        avg_vol = sum(volumes) / len(volumes) if volumes else 0
        recent_vol = sum(volumes[-3:]) / min(3, len(volumes)) if volumes else 0
        vol_ratio = round(recent_vol / avg_vol, 2) if avg_vol > 0 else 1.0

        # RSI label
        if rsi >= 70:
            rsi_label = "overbought"
        elif rsi <= 30:
            rsi_label = "oversold"
        else:
            rsi_label = "neutral"

        return {
            "interval": interval,
            "available": True,
            "rsi": round(rsi, 1),
            "rsi_label": rsi_label,
            "trend": trend,
            "ema_20": round(ema_20, 2),
            "ema_50": round(ema_50, 2),
            "atr_pct": round(atr_pct, 2),
            "vol_ratio": vol_ratio,
            "price": round(current_price, 2),
        }

    async def get_multi_timeframe_analysis(self, coin: str) -> str:
        """
        Fetch candles for 1h and 4h, compute indicators, return a formatted
        summary string ready for LLM consumption.
        """
        timeframes = [("1h", 50), ("4h", 50)]

        async def _fetch_tf(interval, lookback):
            try:
                candles = await self.get_candle_data(coin, interval, lookback)
                return self.compute_technical_summary(candles, interval)
            except Exception as e:
                logger.warning("Candle fetch failed", coin=coin, interval=interval, error=str(e))
                return {"interval": interval, "available": False}

        results = await asyncio.gather(*[_fetch_tf(iv, lb) for iv, lb in timeframes])

        lines = []
        for r in results:
            if not r.get("available"):
                lines.append(f"- {r['interval']}: data unavailable")
                continue
            lines.append(
                f"- {r['interval']}: RSI={r['rsi']} ({r['rsi_label']}), "
                f"trend={r['trend']} (EMA20={r['ema_20']}, EMA50={r['ema_50']}), "
                f"ATR={r['atr_pct']}%, vol={r['vol_ratio']}x avg"
            )

        return "\n".join(lines) if lines else "Multi-timeframe data unavailable."

    async def get_funding_rate(self, asset: str) -> float:
        """Get the current funding rate for an asset (hourly, raw decimal)."""
        coin = asset.replace("-USD", "")
        meta_and_ctxs = await self._info_request("metaAndAssetCtxs")
        meta = meta_and_ctxs[0] if len(meta_and_ctxs) > 0 else {}
        asset_ctxs = meta_and_ctxs[1] if len(meta_and_ctxs) > 1 else []
        universe = meta.get("universe", [])

        for idx, uni in enumerate(universe):
            if uni.get("name") == coin and idx < len(asset_ctxs):
                return float(asset_ctxs[idx].get("funding", 0))
        return 0.0

    # ==================
    # Position Methods
    # ==================

    async def get_positions(self, user_address: str) -> List[Dict[str, Any]]:
        """
        Get all open positions for a user.

        Args:
            user_address: User's wallet address

        Returns:
            List of position dictionaries
        """
        data = await self._info_request("clearinghouseState", {"user": user_address})

        positions = []
        for pos in data.get("assetPositions", []):
            position = pos.get("position", {})
            if float(position.get("szi", 0)) == 0:
                continue  # Skip closed positions

            size = float(position.get("szi", 0))
            entry_price = float(position.get("entryPx", 0))
            unrealized_pnl = float(position.get("unrealizedPnl", 0))

            positions.append({
                "asset": f"{position.get('coin')}-USD",
                "direction": "LONG" if size > 0 else "SHORT",
                "size": abs(size),
                "entry_price": entry_price,
                "unrealized_pnl": unrealized_pnl,
                "margin_used": float(position.get("marginUsed", 0)),
                "liquidation_price": float(position.get("liquidationPx", 0)),
                "leverage": float(position.get("leverage", {}).get("value", 1)),
                "funding_paid": float(position.get("cumFunding", {}).get("sinceChange", 0)),
            })

        return positions

    async def get_account_value(self, user_address: str) -> Dict[str, float]:
        """
        Get account value and margin info.

        Returns:
            Dictionary with account balance, equity, margin used, etc.
        """
        data = await self._info_request("clearinghouseState", {"user": user_address})

        cross_margin = data.get("crossMarginSummary", {})

        return {
            "account_value": float(cross_margin.get("accountValue", 0)),
            "total_margin_used": float(cross_margin.get("totalMarginUsed", 0)),
            "total_ntl_pos": float(cross_margin.get("totalNtlPos", 0)),
            "withdrawable": float(data.get("withdrawable", 0)),
        }

    async def get_clearinghouse_state(self, user_address: str) -> Dict[str, Any]:
        """
        Get full clearinghouse state in a single API call.
        Returns both positions and account value — replaces separate
        get_positions() + get_account_value() calls for PnL sync.
        """
        data = await self._info_request("clearinghouseState", {"user": user_address})

        positions = []
        for pos in data.get("assetPositions", []):
            position = pos.get("position", {})
            if float(position.get("szi", 0)) == 0:
                continue
            size = float(position.get("szi", 0))
            positions.append({
                "asset": f"{position.get('coin')}-USD",
                "direction": "LONG" if size > 0 else "SHORT",
                "size": abs(size),
                "entry_price": float(position.get("entryPx", 0)),
                "unrealized_pnl": float(position.get("unrealizedPnl", 0)),
                "margin_used": float(position.get("marginUsed", 0)),
                "liquidation_price": float(position.get("liquidationPx", 0)),
                "leverage": float(position.get("leverage", {}).get("value", 1)),
                "funding_paid": float(position.get("cumFunding", {}).get("sinceChange", 0)),
            })

        cross_margin = data.get("crossMarginSummary", {})
        account = {
            "account_value": float(cross_margin.get("accountValue", 0)),
            "total_margin_used": float(cross_margin.get("totalMarginUsed", 0)),
            "total_ntl_pos": float(cross_margin.get("totalNtlPos", 0)),
            "withdrawable": float(data.get("withdrawable", 0)),
        }

        return {"positions": positions, "account": account}

    async def get_all_market_prices(self) -> Dict[str, Dict[str, Any]]:
        """
        Fetch all market prices in a single metaAndAssetCtxs call.
        Returns a dict keyed by coin name with price data.
        Used by the Market Data Worker to cache in Redis.
        """
        meta_and_ctxs = await self._info_request("metaAndAssetCtxs")
        meta = meta_and_ctxs[0] if len(meta_and_ctxs) > 0 else {}
        asset_ctxs = meta_and_ctxs[1] if len(meta_and_ctxs) > 1 else []
        universe = meta.get("universe", [])

        prices = {}
        for idx, uni in enumerate(universe):
            coin = uni.get("name")
            if idx < len(asset_ctxs):
                ctx = asset_ctxs[idx]
                prices[coin] = {
                    "markPx": float(ctx.get("markPx", 0)),
                    "funding": float(ctx.get("funding", 0)),
                    "openInterest": float(ctx.get("openInterest", 0)),
                    "dayNtlVlm": float(ctx.get("dayNtlVlm", 0)),
                    "prevDayPx": float(ctx.get("prevDayPx", 0)),
                }

        # Populate asset index cache
        if self._asset_index_cache is None:
            self._asset_index_cache = {u["name"]: i for i, u in enumerate(universe)}

        return prices

    # ==================
    # Order Methods
    # ==================

    def build_order_payload(
        self,
        asset: str,
        is_buy: bool,
        size: float,
        price: Optional[float] = None,
        reduce_only: bool = False,
        order_type: str = "limit",
        time_in_force: str = "Gtc",
    ) -> Dict[str, Any]:
        """
        Build an order payload for signing (legacy format for internal use).

        Args:
            asset: Asset symbol (e.g., "ETH-USD")
            is_buy: True for buy/long, False for sell/short
            size: Position size in asset units
            price: Limit price (None for market orders)
            reduce_only: True to only reduce position
            order_type: "limit" or "market"
            time_in_force: "Gtc" (Good till cancel), "Ioc" (Immediate or cancel), "Alo" (Add liquidity only)

        Returns:
            Order payload ready for signing
        """
        coin = asset.replace("-USD", "")

        if order_type == "market":
            # Market orders use a special price format
            order = {
                "a": self._get_asset_index(coin),  # Asset index
                "b": is_buy,
                "p": "0",  # Market order indicator
                "s": str(size),
                "r": reduce_only,
                "t": {"limit": {"tif": time_in_force}},
            }
        else:
            order = {
                "a": self._get_asset_index(coin),
                "b": is_buy,
                "p": str(price),
                "s": str(size),
                "r": reduce_only,
                "t": {"limit": {"tif": time_in_force}},
            }

        return {
            "type": "order",
            "orders": [order],
            "grouping": "na",
        }

    def build_eip712_order(
        self,
        asset: str,
        is_buy: bool,
        size: float,
        price: Optional[float] = None,
        reduce_only: bool = False,
        order_type: str = "limit",
        nonce: Optional[int] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build proper EIP-712 typed data for Hyperliquid order signing.

        This is the correct format for Turnkey signTypedData.

        Args:
            asset: Asset symbol (e.g., "ETH-USD")
            is_buy: True for buy/long, False for sell/short
            size: Position size in asset units
            price: Limit price (None for market orders)
            reduce_only: True to only reduce position
            order_type: "limit" or "market"
            nonce: Unique nonce (defaults to timestamp in ms)
            client_order_id: Optional client order ID (16 bytes hex)

        Returns:
            Full EIP-712 typed data structure for signing
        """
        coin = asset.replace("-USD", "")
        asset_index = self._get_asset_index(coin)

        # Generate nonce if not provided
        if nonce is None:
            nonce = int(time.time() * 1000)

        # Generate client order ID if not provided (16 bytes)
        if client_order_id is None:
            import secrets
            client_order_id = "0x" + secrets.token_hex(16)

        # Convert price and size to Hyperliquid's fixed-point format
        # Hyperliquid uses 8 decimal places for prices, variable for sizes
        if order_type == "market" and price is not None:
            # IOC market-like order: set worst-case price with 1% slippage
            slippage_mult = 1.01 if is_buy else 0.99
            limit_px = int(price * slippage_mult * 1e8)
        elif price is None:
            limit_px = 0
        else:
            # Convert to integer with 8 decimal places
            limit_px = int(price * 1e8)

        # Size in asset's smallest unit (depends on asset)
        sz = int(size * 1e8)  # 8 decimal places for size

        # Order type: 0 = limit, 1 = IOC, 2 = ALO
        order_type_int = 0  # Default to limit
        if order_type == "market":
            order_type_int = 1  # IOC for market-like behavior

        # Build the domain - use mainnet or testnet chain ID
        domain = {
            **HYPERLIQUID_EIP712_DOMAIN,
            "chainId": 1 if self.is_mainnet else 1337,
        }

        # Build the order message
        order_message = {
            "asset": asset_index,
            "isBuy": is_buy,
            "limitPx": limit_px,
            "sz": sz,
            "reduceOnly": reduce_only,
            "orderType": order_type_int,
            "cloid": client_order_id,
        }

        # Build the full action message
        action_message = {
            "orders": [order_message],
            "grouping": 0,  # 0 = na, 1 = normalTpsl, 2 = positionTpsl
            "nonce": nonce,
        }

        return {
            "domain": domain,
            "types": ORDER_ACTION_TYPES,
            "primaryType": "OrderAction",
            "message": action_message,
            # Include raw values for easy submission later
            "_raw": {
                "asset": asset,
                "coin": coin,
                "is_buy": is_buy,
                "size": size,
                "price": price,
                "reduce_only": reduce_only,
                "order_type": order_type,
                "nonce": nonce,
                "cloid": client_order_id,
            },
        }

    def build_eip712_agent_approval(
        self,
        agent_address: str,
        agent_name: str = "MoleApp Trading Agent",
        nonce: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Build EIP-712 typed data for agent approval signing.

        This is the correct format for approving MoleApp to trade on behalf of a user.

        Args:
            agent_address: MoleApp's agent wallet address
            agent_name: Display name for the agent
            nonce: Unique nonce (defaults to timestamp in ms)

        Returns:
            Full EIP-712 typed data structure for signing
        """
        if nonce is None:
            nonce = int(time.time() * 1000)

        # Build the domain
        domain = {
            **HYPERLIQUID_EIP712_DOMAIN,
            "chainId": 1 if self.is_mainnet else 1337,
        }

        # Build the approval message
        approval_message = {
            "hyperliquidChain": "Mainnet" if self.is_mainnet else "Testnet",
            "agentAddress": agent_address,
            "agentName": agent_name,
            "nonce": nonce,
        }

        return {
            "domain": domain,
            "types": AGENT_APPROVAL_TYPES,
            "primaryType": "ApproveAgent",
            "message": approval_message,
            "_raw": {
                "agent_address": agent_address,
                "agent_name": agent_name,
                "nonce": nonce,
            },
        }

    # Hardcoded fallback for common assets (used if meta fetch fails)
    _FALLBACK_ASSET_INDICES = {
        "BTC": 0, "ETH": 1, "SOL": 2, "DOGE": 3, "AVAX": 4,
        "MATIC": 5, "ARB": 6, "OP": 7, "LINK": 8, "UNI": 9,
    }

    async def _ensure_asset_index_cache(self) -> None:
        """Fetch asset indices from HL meta API and cache them."""
        if self._asset_index_cache is not None:
            return
        try:
            meta = await self._info_request("meta")
            universe = meta.get("universe", [])
            self._asset_index_cache = {
                asset["name"]: idx for idx, asset in enumerate(universe)
            }
            logger.info("Asset index cache populated", count=len(self._asset_index_cache))
        except Exception as e:
            logger.warning("Failed to fetch asset indices from meta, using fallback", error=str(e))
            self._asset_index_cache = dict(self._FALLBACK_ASSET_INDICES)

    async def get_asset_index(self, coin: str) -> int:
        """Get the asset index for an asset symbol (cached from HL meta API)."""
        await self._ensure_asset_index_cache()
        idx = self._asset_index_cache.get(coin)
        if idx is not None:
            return idx
        # Fallback for unknown assets
        return self._FALLBACK_ASSET_INDICES.get(coin, 0)

    def _get_asset_index(self, coin: str) -> int:
        """Synchronous fallback — prefer async get_asset_index() when possible."""
        if self._asset_index_cache:
            return self._asset_index_cache.get(coin, 0)
        return self._FALLBACK_ASSET_INDICES.get(coin, 0)

    async def place_order(
        self,
        signed_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Place an order using a pre-signed payload.

        Args:
            signed_payload: Order payload with signature (can be legacy or EIP-712 format)

        Returns:
            Order response from Hyperliquid
        """
        try:
            # Handle both legacy and EIP-712 signed payloads
            if "signedPayload" in signed_payload:
                # Legacy format from wallet-service
                action = signed_payload["signedPayload"].get("action", signed_payload.get("action"))
                signature = signed_payload["signedPayload"].get("signature", signed_payload.get("signature"))
                nonce = signed_payload["signedPayload"].get("nonce", signed_payload.get("nonce"))
            else:
                action = signed_payload.get("action")
                signature = signed_payload.get("signature")
                nonce = signed_payload.get("nonce")

            result = await self._exchange_request(
                action=action,
                signature=signature,
                nonce=nonce,
            )

            logger.info(
                "Order placed",
                status=result.get("status"),
                response=result.get("response"),
            )

            return {
                "success": result.get("status") == "ok",
                "order_id": result.get("response", {}).get("data", {}).get("statuses", [{}])[0].get("resting", {}).get("oid"),
                "tx_hash": result.get("response", {}).get("data", {}).get("statuses", [{}])[0].get("filled", {}).get("oid"),
                "response": result,
            }

        except Exception as e:
            logger.error("Order placement failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
            }

    async def submit_withdrawal(
        self,
        signed_withdrawal: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Submit a signed withdrawal to Hyperliquid's exchange endpoint.

        The withdrawal must already be signed by the MASTER address (user's Turnkey EOA)
        via wallet-service. The signed payload should contain:
        - withdrawalPayload.action: { type: 'withdraw3', ... }
        - withdrawalPayload.nonce: timestamp
        - withdrawalPayload.signature: hex string

        After submission, Hyperliquid bridges USDC back to the destination
        address on Arbitrum automatically.

        Args:
            signed_withdrawal: Response from wallet-service withdraw endpoint

        Returns:
            Result dictionary with success status
        """
        try:
            payload = signed_withdrawal.get("withdrawalPayload", signed_withdrawal)
            action = payload.get("action")
            signature = payload.get("signature")
            nonce = payload.get("nonce")

            if not all([action, signature, nonce]):
                return {"success": False, "error": "Invalid withdrawal payload: missing action, signature, or nonce"}

            result = await self._exchange_request(
                action=action,
                signature=signature,
                nonce=nonce,
            )

            logger.info(
                "Withdrawal submitted to Hyperliquid",
                status=result.get("status"),
            )

            return {
                "success": result.get("status") == "ok",
                "response": result,
            }

        except Exception as e:
            logger.error("Withdrawal submission failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
            }

    async def treasury_send_usd(
        self,
        destination: str,
        amount: float,
    ) -> Dict[str, Any]:
        """
        Transfer USDC on HL L1 from the pre-funded treasury wallet to a user address.

        Uses local signing with HL_TREASURY_PRIVATE_KEY (no Turnkey needed).
        Only for testnet auto-funding when the Arbitrum bridge is not available.
        BLOCKED on mainnet — users must deposit via the Arbitrum bridge instead.

        Args:
            destination: User's HL L1 address (signer EOA)
            amount: USDC amount to transfer

        Returns:
            Result dict with success status
        """
        if self.is_mainnet:
            return {"success": False, "error": "Treasury auto-funding is only available on testnet. Use the Arbitrum bridge for mainnet deposits."}

        import msgpack
        from eth_account import Account
        from eth_account.messages import encode_structured_data
        from eth_hash.auto import keccak

        from app.config import get_settings

        settings = get_settings()

        if not settings.hl_treasury_private_key:
            return {"success": False, "error": "HL_TREASURY_PRIVATE_KEY not configured"}

        timestamp = int(time.time() * 1000)

        # Build the usdSend action
        action = {
            "type": "usdSend",
            "hyperliquidChain": "Mainnet" if self.is_mainnet else "Testnet",
            "signatureChainId": "0xa4b1" if self.is_mainnet else "0x66eee",
            "destination": destination,
            "amount": str(amount),
            "time": timestamp,
        }

        # Phantom agent signing: hash(msgpack(action) + nonce_bytes + b"\x00")
        packed = msgpack.packb(action)
        packed += timestamp.to_bytes(8, "big")
        packed += b"\x00"  # not a vault action
        connection_id = keccak(packed)

        # EIP-712 typed data for the phantom agent
        typed_data = {
            "types": {
                "EIP712Domain": [
                    {"name": "name", "type": "string"},
                    {"name": "version", "type": "string"},
                    {"name": "chainId", "type": "uint256"},
                    {"name": "verifyingContract", "type": "address"},
                ],
                "Agent": [
                    {"name": "source", "type": "string"},
                    {"name": "connectionId", "type": "bytes32"},
                ],
            },
            "primaryType": "Agent",
            "domain": {
                "name": "Exchange",
                "version": "1",
                "chainId": 1 if self.is_mainnet else 1337,
                "verifyingContract": "0x0000000000000000000000000000000000000000",
            },
            "message": {
                "source": "a" if self.is_mainnet else "b",
                "connectionId": connection_id,
            },
        }

        # Sign with treasury private key
        account = Account.from_key(settings.hl_treasury_private_key)
        signable = encode_structured_data(typed_data)
        signed = account.sign_message(signable)

        signature = {
            "r": hex(signed.r),
            "s": hex(signed.s),
            "v": signed.v,
        }

        logger.info(
            "Submitting treasury usdSend",
            destination=destination,
            amount=amount,
            treasury=settings.hl_treasury_address,
        )

        try:
            result = await self._exchange_request(
                action=action,
                signature=signature,
                nonce=timestamp,
            )

            success = result.get("status") == "ok"
            logger.info(
                "Treasury usdSend result",
                success=success,
                response=result,
            )

            return {"success": success, "response": result}

        except Exception as e:
            logger.error("Treasury usdSend failed", error=str(e))
            return {"success": False, "error": str(e)}

    async def place_order_eip712(
        self,
        typed_data: Dict[str, Any],
        signature: str,
    ) -> Dict[str, Any]:
        """
        Place an order using EIP-712 typed data and signature.

        This is the proper method for orders signed via Turnkey signTypedData.

        Args:
            typed_data: The EIP-712 typed data that was signed
            signature: The signature from Turnkey (hex string with 0x prefix)

        Returns:
            Order response from Hyperliquid
        """
        try:
            # Extract raw values from typed data
            raw = typed_data.get("_raw", {})
            message = typed_data.get("message", {})

            # Parse signature into r, s, v components
            sig = signature.replace("0x", "")
            r = "0x" + sig[:64]
            s = "0x" + sig[64:128]
            v = int(sig[128:130], 16) if len(sig) >= 130 else 27

            # Build the action for Hyperliquid's exchange endpoint
            orders = message.get("orders", [])
            hl_orders = []
            for order in orders:
                # Respect orderType for time-in-force
                order_type_int = order.get("orderType", 0)
                if order_type_int == 1:
                    tif_obj = {"limit": {"tif": "Ioc"}}  # IOC (market-like)
                elif order_type_int == 2:
                    tif_obj = {"limit": {"tif": "Alo"}}  # Add-Liquidity-Only
                else:
                    tif_obj = {"limit": {"tif": "Gtc"}}  # Default GTC

                hl_orders.append({
                    "a": order["asset"],
                    "b": order["isBuy"],
                    "p": str(order["limitPx"]),
                    "s": str(order["sz"]),
                    "r": order.get("reduceOnly", False),
                    "t": tif_obj,
                })

            action = {
                "type": "order",
                "orders": hl_orders,
                "grouping": "na",
            }

            # Format signature for Hyperliquid
            formatted_signature = {
                "r": r,
                "s": s,
                "v": v,
            }

            result = await self._exchange_request(
                action=action,
                signature=formatted_signature,
                nonce=message.get("nonce"),
            )

            logger.info(
                "EIP-712 order placed",
                status=result.get("status"),
                asset=raw.get("asset"),
            )

            return {
                "success": result.get("status") == "ok",
                "order_id": result.get("response", {}).get("data", {}).get("statuses", [{}])[0].get("resting", {}).get("oid"),
                "tx_hash": result.get("response", {}).get("data", {}).get("statuses", [{}])[0].get("filled", {}).get("oid"),
                "response": result,
            }

        except Exception as e:
            logger.error("EIP-712 order placement failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
            }

    async def cancel_order(
        self,
        signed_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Cancel an order using a pre-signed payload."""
        try:
            result = await self._exchange_request(
                action=signed_payload["action"],
                signature=signed_payload["signature"],
                nonce=signed_payload["nonce"],
            )

            return {
                "success": result.get("status") == "ok",
                "response": result,
            }

        except Exception as e:
            logger.error("Order cancellation failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
            }

    # ==================
    # Agent Approval Methods
    # ==================

    async def check_agent_approval(
        self,
        user_address: str,
        agent_address: str,
    ) -> bool:
        """
        Check if an agent is approved to trade on behalf of a user.

        Args:
            user_address: User's wallet address
            agent_address: Agent's wallet address

        Returns:
            True if agent is approved
        """
        try:
            # Get user's approved agents
            data = await self._info_request(
                "extraAgents",
                {"user": user_address},
            )

            for agent in data.get("agents", []):
                if agent.get("address", "").lower() == agent_address.lower():
                    return True

            return False

        except Exception as e:
            logger.error(
                "Agent approval check failed",
                user=user_address,
                agent=agent_address,
                error=str(e),
            )
            return False

    def build_agent_approval_payload(
        self,
        agent_address: str,
        name: str = "MoleApp Agent",
    ) -> Dict[str, Any]:
        """
        Build the payload for approving an agent.

        This needs to be signed by the user's Turnkey wallet.
        """
        return {
            "type": "approveAgent",
            "hyperliquidChain": "Mainnet" if self.is_mainnet else "Testnet",
            "agentAddress": agent_address,
            "agentName": name,
            "nonce": int(datetime.now().timestamp() * 1000),
        }

    def build_agent_approval_for_signing(
        self,
        agent_address: str,
        agent_name: str = "MoleApp Trading Agent",
    ) -> tuple:
        """
        Build the approveAgent action AND the phantom agent EIP-712 typed data
        for signing by the user's master key.

        Returns:
            (action, typed_data, nonce) — action for HL exchange,
            typed_data for wallet-service to sign, nonce for submission.
        """
        import msgpack
        from eth_hash.auto import keccak

        nonce = int(time.time() * 1000)

        action = {
            "type": "approveAgent",
            "hyperliquidChain": "Mainnet" if self.is_mainnet else "Testnet",
            "signatureChainId": "0xa4b1" if self.is_mainnet else "0x66eee",
            "agentAddress": agent_address,
            "agentName": agent_name,
            "nonce": nonce,
        }

        # Phantom agent hash: keccak(msgpack(action) + nonce_bytes + b"\x00")
        packed = msgpack.packb(action)
        packed += nonce.to_bytes(8, "big")
        packed += b"\x00"
        connection_id = keccak(packed)

        typed_data = {
            "domain": {
                "name": "Exchange",
                "version": "1",
                "chainId": 1 if self.is_mainnet else 1337,
                "verifyingContract": "0x0000000000000000000000000000000000000000",
            },
            "types": {
                "Agent": [
                    {"name": "source", "type": "string"},
                    {"name": "connectionId", "type": "bytes32"},
                ],
            },
            "primaryType": "Agent",
            "message": {
                "source": "a" if self.is_mainnet else "b",
                "connectionId": "0x" + connection_id.hex(),
            },
        }

        return action, typed_data, nonce

    async def submit_agent_approval(
        self,
        signed_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Submit a signed agent approval to Hyperliquid's exchange endpoint.

        The approval must already be signed (EIP-712 phantom agent) by the
        user's master EOA via wallet-service.

        Args:
            signed_payload: Dict with action, nonce, and signature (hex string)

        Returns:
            Result dict with success status
        """
        try:
            action = signed_payload.get("action")
            signature_hex = signed_payload.get("signature", "")
            nonce = signed_payload.get("nonce")

            if not all([action, signature_hex, nonce]):
                return {
                    "success": False,
                    "error": "Invalid approval payload: missing action, signature, or nonce",
                }

            # Parse hex signature into {r, s, v} for HL exchange API
            sig = signature_hex.replace("0x", "")
            r = "0x" + sig[:64]
            s = "0x" + sig[64:128]
            v = int(sig[128:130], 16) if len(sig) >= 130 else 27

            formatted_signature = {"r": r, "s": s, "v": v}

            result = await self._exchange_request(
                action=action,
                signature=formatted_signature,
                nonce=nonce,
            )

            logger.info(
                "Agent approval submitted to HL",
                status=result.get("status"),
            )

            return {
                "success": result.get("status") == "ok",
                "response": result,
            }

        except Exception as e:
            logger.error("Agent approval submission failed", error=str(e))
            return {"success": False, "error": str(e)}

    # ==================
    # Utility Methods
    # ==================

    async def get_user_fills(
        self,
        user_address: str,
        start_time: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get historical fills for a user."""
        data = await self._info_request(
            "userFills",
            {
                "user": user_address,
                "startTime": start_time or 0,
            },
        )
        return data

    async def get_open_orders(self, user_address: str) -> List[Dict[str, Any]]:
        """Get all open orders for a user."""
        data = await self._info_request("openOrders", {"user": user_address})
        return data

    async def get_recent_trades(self, asset: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trades for an asset."""
        data = await self._info_request(
            "recentTrades",
            {"coin": asset.replace("-USD", "")},
        )
        return data[:limit]

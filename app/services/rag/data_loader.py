"""
Market Data Loader for FAISS Index
Fetches historical market data and generates patterns for the vector store
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class MarketDataLoader:
    """
    Loads and processes historical market data to generate patterns for FAISS.

    Data sources:
    - CoinGecko API (free tier) for historical prices
    - Binance API for OHLCV data
    """

    def __init__(self, data_dir: str = "/app/data/raw"):
        """Initialize the market data loader."""
        self.data_dir = data_dir
        self.coingecko_base_url = "https://api.coingecko.com/api/v3"
        self.binance_base_url = "https://api.binance.com/api/v3"

        # Asset mapping for different APIs
        self.asset_mapping = {
            "ETH-USD": {"coingecko": "ethereum", "binance": "ETHUSDT"},
            "BTC-USD": {"coingecko": "bitcoin", "binance": "BTCUSDT"},
        }

    async def fetch_historical_data(
        self,
        asset: str,
        days: int = 365,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for an asset.

        Args:
            asset: Asset symbol (e.g., "ETH-USD")
            days: Number of days of history
            interval: Data interval

        Returns:
            DataFrame with OHLCV data
        """
        mapping = self.asset_mapping.get(asset)
        if not mapping:
            raise ValueError(f"Unknown asset: {asset}")

        try:
            # Try Binance first (more reliable)
            df = await self._fetch_from_binance(
                symbol=mapping["binance"],
                days=days,
                interval=interval,
            )
            logger.info(
                "Fetched data from Binance",
                asset=asset,
                rows=len(df),
            )
            return df
        except Exception as e:
            logger.warning(
                "Binance fetch failed, trying CoinGecko",
                asset=asset,
                error=str(e),
            )

        # Fallback to CoinGecko
        df = await self._fetch_from_coingecko(
            coin_id=mapping["coingecko"],
            days=days,
        )
        logger.info(
            "Fetched data from CoinGecko",
            asset=asset,
            rows=len(df),
        )
        return df

    async def _fetch_from_binance(
        self,
        symbol: str,
        days: int,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Binance."""
        async with httpx.AsyncClient() as client:
            # Binance klines endpoint
            end_time = int(datetime.now().timestamp() * 1000)
            start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

            response = await client.get(
                f"{self.binance_base_url}/klines",
                params={
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": start_time,
                    "endTime": end_time,
                    "limit": 1000,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

        # Parse Binance klines format
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore",
        ])

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)

        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)

        return df

    async def _fetch_from_coingecko(
        self,
        coin_id: str,
        days: int,
    ) -> pd.DataFrame:
        """Fetch historical prices from CoinGecko."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.coingecko_base_url}/coins/{coin_id}/market_chart",
                params={
                    "vs_currency": "usd",
                    "days": days,
                    "interval": "daily",
                },
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

        # Parse CoinGecko format (only has prices, not full OHLCV)
        prices = data.get("prices", [])
        df = pd.DataFrame(prices, columns=["timestamp", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        # Approximate OHLCV from daily close prices
        df["open"] = df["close"].shift(1)
        df["high"] = df["close"] * 1.02  # Approximate
        df["low"] = df["close"] * 0.98   # Approximate
        df["volume"] = 0  # Not available from CoinGecko free tier

        df = df.dropna()
        return df[["open", "high", "low", "close", "volume"]]

    def calculate_metrics(self, df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
        """
        Calculate trading metrics from OHLCV data.

        Args:
            df: DataFrame with OHLCV data
            window: Rolling window for calculations

        Returns:
            DataFrame with additional metric columns
        """
        df = df.copy()

        # Returns
        df["return_1d"] = df["close"].pct_change()
        df[f"return_{window}d"] = df["close"].pct_change(window)

        # Volatility (standard deviation of returns)
        df[f"volatility_{window}d"] = df["return_1d"].rolling(window).std()

        # Maximum drawdown over window
        df[f"max_drawdown_{window}d"] = self._calculate_rolling_max_drawdown(
            df["close"], window
        )

        # Moving averages
        df["sma_20"] = df["close"].rolling(20).mean()
        df["sma_50"] = df["close"].rolling(50).mean()

        # RSI
        df["rsi_14"] = self._calculate_rsi(df["close"], 14)

        # Volume trends
        df["volume_sma_20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"]

        return df

    def _calculate_rolling_max_drawdown(
        self,
        prices: pd.Series,
        window: int,
    ) -> pd.Series:
        """Calculate rolling maximum drawdown."""
        def max_dd(x):
            if len(x) < 2:
                return 0
            peak = x.cummax()
            drawdown = (x - peak) / peak
            return drawdown.min()

        return prices.rolling(window).apply(max_dd, raw=False)

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def extract_patterns(
        self,
        df: pd.DataFrame,
        asset: str,
        pattern_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract significant market patterns from historical data.

        Args:
            df: DataFrame with OHLCV and metrics
            asset: Asset symbol
            pattern_types: Types of patterns to extract

        Returns:
            List of pattern dictionaries for FAISS indexing
        """
        patterns = []
        pattern_types = pattern_types or [
            "accumulation",
            "distribution",
            "breakout",
            "consolidation",
            "trend_continuation",
        ]

        # Ensure metrics are calculated
        if "return_30d" not in df.columns:
            df = self.calculate_metrics(df)

        # Drop rows with NaN metrics
        df = df.dropna(subset=["return_30d", "volatility_30d", "max_drawdown_30d"])

        for idx, row in df.iterrows():
            pattern_type = self._classify_pattern(row, df.loc[:idx])

            if pattern_type not in pattern_types:
                continue

            # Generate context description
            context = self._generate_pattern_context(
                date=idx,
                asset=asset,
                pattern_type=pattern_type,
                row=row,
            )

            # Look ahead for outcome (if available)
            outcome = self._get_outcome(df, idx, lookahead=30)

            pattern = {
                "id": f"pattern_{asset}_{idx.strftime('%Y%m%d')}",
                "date": idx.isoformat() if hasattr(idx, "isoformat") else str(idx),
                "asset": asset,
                "pattern_type": pattern_type,
                "context": context,
                "metrics": {
                    "return_30d": float(row.get("return_30d", 0)),
                    "volatility_30d": float(row.get("volatility_30d", 0)),
                    "max_drawdown_30d": float(row.get("max_drawdown_30d", 0)),
                    "rsi_14": float(row.get("rsi_14", 50)),
                    "price": float(row["close"]),
                    "volume_ratio": float(row.get("volume_ratio", 1)),
                },
                "outcome": outcome,
            }
            patterns.append(pattern)

        logger.info(
            "Extracted patterns",
            asset=asset,
            total_patterns=len(patterns),
            pattern_types=list(set(p["pattern_type"] for p in patterns)),
        )
        return patterns

    def _classify_pattern(self, row: pd.Series, history: pd.DataFrame) -> str:
        """Classify the current market state into a pattern type."""
        rsi = row.get("rsi_14", 50)
        return_30d = row.get("return_30d", 0)
        volatility = row.get("volatility_30d", 0)

        # Simple pattern classification
        if rsi < 30 and return_30d < -0.15:
            return "accumulation"
        elif rsi > 70 and return_30d > 0.15:
            return "distribution"
        elif return_30d > 0.20 and volatility > 0.04:
            return "breakout"
        elif abs(return_30d) < 0.05 and volatility < 0.02:
            return "consolidation"
        elif return_30d > 0.10:
            return "trend_continuation"
        else:
            return "neutral"

    def _generate_pattern_context(
        self,
        date: datetime,
        asset: str,
        pattern_type: str,
        row: pd.Series,
    ) -> str:
        """Generate a natural language context description for the pattern."""
        price = row["close"]
        return_30d = row.get("return_30d", 0) * 100
        volatility = row.get("volatility_30d", 0) * 100
        rsi = row.get("rsi_14", 50)
        max_dd = row.get("max_drawdown_30d", 0) * 100

        date_str = date.strftime("%B %Y") if hasattr(date, "strftime") else str(date)

        context = f"""
{asset} {pattern_type} pattern observed in {date_str}.
Price: ${price:,.2f}
30-day return: {return_30d:+.1f}%
30-day volatility: {volatility:.1f}%
Maximum drawdown (30d): {max_dd:.1f}%
RSI(14): {rsi:.0f}

Market conditions: {"oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral"}
territory with {"high" if volatility > 4 else "moderate" if volatility > 2 else "low"} volatility.
        """.strip()

        return context

    def _get_outcome(
        self,
        df: pd.DataFrame,
        current_idx,
        lookahead: int = 30,
    ) -> Dict[str, Any]:
        """Get the outcome data for a pattern (what happened next)."""
        try:
            # Find the index position
            pos = df.index.get_loc(current_idx)

            # Look ahead
            if pos + lookahead >= len(df):
                return {}

            future_prices = df.iloc[pos:pos + lookahead + 1]["close"]
            current_price = future_prices.iloc[0]
            final_price = future_prices.iloc[-1]

            # Calculate outcome metrics
            return_next = (final_price - current_price) / current_price

            # Max drawdown in the lookahead period
            peak = future_prices.cummax()
            drawdown = (future_prices - peak) / peak
            max_dd_next = drawdown.min()

            return {
                "return_next_30d": float(return_next),
                "max_drawdown_next_30d": float(max_dd_next),
                "final_price": float(final_price),
            }
        except Exception:
            return {}

    async def build_full_index(
        self,
        assets: List[str],
        days: int = 365,
    ) -> List[Dict[str, Any]]:
        """
        Build a complete pattern index from historical data.

        Args:
            assets: List of assets to index
            days: Days of history to fetch

        Returns:
            List of all patterns ready for FAISS indexing
        """
        all_patterns = []

        for asset in assets:
            try:
                logger.info("Processing asset", asset=asset, days=days)

                # Fetch data
                df = await self.fetch_historical_data(asset, days=days)

                # Calculate metrics
                df = self.calculate_metrics(df)

                # Extract patterns
                patterns = self.extract_patterns(df, asset)

                all_patterns.extend(patterns)

                # Small delay to respect rate limits
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(
                    "Failed to process asset",
                    asset=asset,
                    error=str(e),
                )
                continue

        logger.info(
            "Full index built",
            total_patterns=len(all_patterns),
            assets=assets,
        )
        return all_patterns

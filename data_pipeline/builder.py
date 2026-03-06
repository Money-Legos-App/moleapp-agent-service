#!/usr/bin/env python3
"""
FAISS Index Builder
Builds the initial FAISS index from historical market data.

Usage:
    python -m data_pipeline.builder --days 365 --assets ETH-USD,BTC-USD
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


async def main(days: int, assets: list[str], output_path: str):
    """Build the FAISS index."""
    from app.services.rag import FAISSStore, MarketDataLoader

    print(f"Building FAISS index...")
    print(f"  Days of history: {days}")
    print(f"  Assets: {assets}")
    print(f"  Output path: {output_path}")
    print()

    # Initialize data loader
    data_loader = MarketDataLoader()

    # Fetch and process data for all assets
    print("Fetching historical data...")
    all_patterns = await data_loader.build_full_index(
        assets=assets,
        days=days,
    )

    print(f"Total patterns extracted: {len(all_patterns)}")

    # Initialize FAISS store
    faiss_store = FAISSStore(index_path=output_path)
    await faiss_store.initialize()

    # Add patterns to index
    print("Adding patterns to FAISS index...")
    added = await faiss_store.add_patterns(all_patterns)

    print(f"Patterns added: {added}")

    # Print stats
    stats = faiss_store.get_stats()
    print()
    print("Index Statistics:")
    print(f"  Vectors: {stats['num_vectors']}")
    print(f"  Patterns: {stats['num_patterns']}")
    print(f"  Assets: {stats['assets']}")
    print(f"  Index path: {stats['index_path']}")

    print()
    print("FAISS index built successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index from historical market data")
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of days of historical data to fetch (default: 365)",
    )
    parser.add_argument(
        "--assets",
        type=str,
        default="ETH-USD,BTC-USD",
        help="Comma-separated list of assets (default: ETH-USD,BTC-USD)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/app/data/faiss_index",
        help="Output path for FAISS index (default: /app/data/faiss_index)",
    )

    args = parser.parse_args()
    assets = [a.strip() for a in args.assets.split(",")]

    asyncio.run(main(
        days=args.days,
        assets=assets,
        output_path=args.output,
    ))

"""
FAISS Vector Store for Market Pattern Retrieval
Stores historical market patterns and enables semantic similarity search
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import structlog
from sentence_transformers import SentenceTransformer

logger = structlog.get_logger(__name__)


class FAISSStore:
    """
    FAISS-based vector store for storing and retrieving market patterns.

    Each pattern includes:
    - Contextual description of market conditions
    - Historical metrics (returns, volatility, drawdowns)
    - Outcome data for the following period
    """

    def __init__(
        self,
        index_path: Optional[str] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Initialize FAISS store.

        Args:
            index_path: Path to store/load FAISS index
            embedding_model: HuggingFace model for text embeddings
        """
        from app.config import get_settings

        settings = get_settings()
        self.index_path = Path(index_path or settings.faiss_index_path)
        self.embedding_model_name = embedding_model or settings.embedding_model

        self._encoder: Optional[SentenceTransformer] = None
        self._index = None
        self._metadata: List[Dict[str, Any]] = []
        self._is_ready = False

        logger.info(
            "FAISSStore initialized",
            index_path=str(self.index_path),
            embedding_model=self.embedding_model_name,
        )

    @property
    def is_ready(self) -> bool:
        """Check if the store is ready for queries."""
        return self._is_ready

    @property
    def encoder(self) -> SentenceTransformer:
        """Lazy-load the sentence transformer encoder."""
        if self._encoder is None:
            logger.info("Loading embedding model", model=self.embedding_model_name)
            self._encoder = SentenceTransformer(self.embedding_model_name)
        return self._encoder

    async def initialize(self) -> None:
        """Initialize the FAISS store, loading existing index if available."""
        try:
            import faiss

            index_file = self.index_path / "index.bin"
            metadata_file = self.index_path / "metadata.json"

            if index_file.exists() and metadata_file.exists():
                # Load existing index
                logger.info("Loading existing FAISS index", path=str(index_file))
                self._index = faiss.read_index(str(index_file))

                with open(metadata_file, "r") as f:
                    self._metadata = json.load(f)

                logger.info(
                    "FAISS index loaded",
                    num_vectors=self._index.ntotal,
                    num_patterns=len(self._metadata),
                )
                self._is_ready = True
            else:
                # Initialize empty index with default dimension
                # Don't load the encoder yet - that can block startup while downloading model
                logger.info("No existing index found, creating empty index")

                # Use default dimension for all-MiniLM-L6-v2 (384)
                # This will be validated when the encoder is first used
                default_dimension = 384

                # Create FAISS index (using L2 distance, can switch to cosine)
                self._index = faiss.IndexFlatL2(default_dimension)
                self._metadata = []

                # Ensure directory exists
                self.index_path.mkdir(parents=True, exist_ok=True)

                # Mark as ready even with empty index
                self._is_ready = True
                logger.info("Empty FAISS index created", dimension=default_dimension)

        except Exception as e:
            logger.error("Failed to initialize FAISS store", error=str(e))
            self._is_ready = False
            # Don't raise - allow service to start without FAISS
            # This enables health checks to report the issue

    async def add_patterns(
        self,
        patterns: List[Dict[str, Any]],
        save: bool = True,
    ) -> int:
        """
        Add market patterns to the index.

        Args:
            patterns: List of pattern dictionaries with 'context' field for embedding
            save: Whether to save the index after adding

        Returns:
            Number of patterns added
        """
        import faiss

        if not self._is_ready:
            raise RuntimeError("FAISS store not initialized")

        if not patterns:
            return 0

        # Extract contexts for embedding
        contexts = [p.get("context", "") for p in patterns]

        # Generate embeddings
        logger.info("Generating embeddings", num_patterns=len(patterns))
        embeddings = self.encoder.encode(contexts, show_progress_bar=False)
        embeddings = np.array(embeddings).astype("float32")

        # Add to FAISS index
        self._index.add(embeddings)

        # Store metadata
        for pattern in patterns:
            self._metadata.append({
                "id": pattern.get("id", f"pattern_{len(self._metadata)}"),
                "date": pattern.get("date"),
                "asset": pattern.get("asset"),
                "pattern_type": pattern.get("pattern_type"),
                "context": pattern.get("context"),
                "metrics": pattern.get("metrics", {}),
                "outcome": pattern.get("outcome", {}),
            })

        if save:
            await self._save_index()

        logger.info(
            "Patterns added to index",
            added=len(patterns),
            total=self._index.ntotal,
        )
        return len(patterns)

    async def search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar market patterns.

        Args:
            query: Text query describing current market conditions
            k: Number of results to return
            filters: Optional filters (e.g., {"asset": "ETH-USD"})

        Returns:
            List of matching patterns with similarity scores
        """
        if not self._is_ready or self._index.ntotal == 0:
            logger.warning("FAISS index is empty or not ready")
            return []

        # Generate query embedding
        query_embedding = self.encoder.encode([query])[0]
        query_embedding = np.array([query_embedding]).astype("float32")

        # Search FAISS index (get more results for filtering)
        search_k = min(k * 3, self._index.ntotal) if filters else k
        distances, indices = self._index.search(query_embedding, search_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue

            pattern = self._metadata[idx].copy()

            # Apply filters
            if filters:
                match = True
                for key, value in filters.items():
                    if pattern.get(key) != value:
                        match = False
                        break
                if not match:
                    continue

            # Convert L2 distance to similarity score (0-1)
            # Lower distance = higher similarity
            similarity = 1.0 / (1.0 + float(dist))
            pattern["similarity_score"] = similarity
            pattern["distance"] = float(dist)

            results.append(pattern)

            if len(results) >= k:
                break

        logger.debug(
            "Search completed",
            query_length=len(query),
            results=len(results),
        )
        return results

    async def get_risk_context(self, asset: str) -> Dict[str, float]:
        """
        Get aggregated risk metrics from historical patterns for an asset.

        Args:
            asset: Asset symbol (e.g., "ETH-USD")

        Returns:
            Dictionary with risk metrics
        """
        if not self._is_ready:
            return {"max_drawdown_30d": -0.20, "volatility_30d": 0.05}

        # Find all patterns for this asset
        asset_patterns = [
            p for p in self._metadata
            if p.get("asset") == asset
        ]

        if not asset_patterns:
            # Return conservative defaults
            return {
                "max_drawdown_30d": -0.20,
                "avg_drawdown_30d": -0.10,
                "volatility_30d": 0.05,
                "sample_count": 0,
            }

        # Aggregate metrics
        drawdowns = [
            p.get("metrics", {}).get("max_drawdown_30d", -0.10)
            for p in asset_patterns
            if "metrics" in p
        ]
        volatilities = [
            p.get("metrics", {}).get("volatility_30d", 0.05)
            for p in asset_patterns
            if "metrics" in p
        ]

        return {
            "max_drawdown_30d": min(drawdowns) if drawdowns else -0.20,
            "avg_drawdown_30d": sum(drawdowns) / len(drawdowns) if drawdowns else -0.10,
            "volatility_30d": sum(volatilities) / len(volatilities) if volatilities else 0.05,
            "sample_count": len(asset_patterns),
        }

    async def get_pattern_by_id(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific pattern by its ID."""
        for pattern in self._metadata:
            if pattern.get("id") == pattern_id:
                return pattern
        return None

    async def _save_index(self) -> None:
        """Save the FAISS index and metadata to disk."""
        import faiss

        self.index_path.mkdir(parents=True, exist_ok=True)

        index_file = self.index_path / "index.bin"
        metadata_file = self.index_path / "metadata.json"

        # Save FAISS index
        faiss.write_index(self._index, str(index_file))

        # Save metadata
        with open(metadata_file, "w") as f:
            json.dump(self._metadata, f, indent=2, default=str)

        logger.info(
            "FAISS index saved",
            path=str(self.index_path),
            num_vectors=self._index.ntotal,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index."""
        return {
            "is_ready": self._is_ready,
            "num_vectors": self._index.ntotal if self._index else 0,
            "num_patterns": len(self._metadata),
            "index_path": str(self.index_path),
            "embedding_model": self.embedding_model_name,
            "assets": list(set(p.get("asset") for p in self._metadata if p.get("asset"))),
        }

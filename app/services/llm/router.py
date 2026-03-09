"""
LLM Router — The Fork in the Road

Dispatches LLM calls to the correct provider based on mission's llmProvider field.
Both providers receive identical prompts and market data for a fair competition.

Usage:
    router = LLMRouter()
    client = router.get_client("qwen")   # or "deepseek"
    signal = await client.analyze_market(...)
    await router.close()
"""

from typing import Optional, Union

import structlog

from .deepseek_client import DeepSeekClient
from .qwen_client import QwenClient

logger = structlog.get_logger(__name__)

# Type alias for either LLM client (same interface)
LLMClient = Union[DeepSeekClient, QwenClient]

# Valid provider names
VALID_PROVIDERS = {"deepseek", "qwen"}
DEFAULT_PROVIDER = "deepseek"


class LLMRouter:
    """
    Routes LLM calls to the correct provider based on mission assignment.

    Lazily initializes clients — only creates a QwenClient if at least
    one active mission uses Qwen, avoiding unnecessary API connections.
    """

    def __init__(self):
        self._clients: dict[str, LLMClient] = {}

    def get_client(self, provider: Optional[str] = None) -> LLMClient:
        """
        Get the LLM client for the specified provider.

        Args:
            provider: "deepseek" or "qwen". Defaults to "deepseek".

        Returns:
            The appropriate LLM client instance.
        """
        provider = (provider or DEFAULT_PROVIDER).lower().strip()

        if provider not in VALID_PROVIDERS:
            logger.warning(
                "Unknown LLM provider, falling back to deepseek",
                requested=provider,
            )
            provider = DEFAULT_PROVIDER

        if provider not in self._clients:
            if provider == "qwen":
                self._clients[provider] = QwenClient()
            else:
                self._clients[provider] = DeepSeekClient()

            logger.info("LLM client initialized", provider=provider)

        return self._clients[provider]

    async def close(self) -> None:
        """Close all initialized LLM clients."""
        for provider, client in self._clients.items():
            try:
                await client.close()
            except Exception as e:
                logger.warning("Error closing LLM client", provider=provider, error=str(e))
        self._clients.clear()

    @property
    def active_providers(self) -> list[str]:
        """List of currently initialized provider names."""
        return list(self._clients.keys())

"""
Langfuse Observability Client — Singleton

Provides trace, span, and generation wrappers with safety guards.
All methods are no-ops when Langfuse is disabled or misconfigured.
Langfuse NEVER raises — every public method is wrapped in try/except.
"""

from functools import lru_cache
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


class _NoOpTrace:
    """Returned when Langfuse is disabled. All methods are safe no-ops."""

    def span(self, **kwargs) -> "_NoOpTrace":
        return self

    def generation(self, **kwargs) -> "_NoOpTrace":
        return self

    def event(self, **kwargs) -> "_NoOpTrace":
        return self

    def score(self, **kwargs) -> "_NoOpTrace":
        return self

    def update(self, **kwargs) -> "_NoOpTrace":
        return self

    def end(self, **kwargs) -> "_NoOpTrace":
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# Singleton no-op instance
_NOOP = _NoOpTrace()


class LangfuseClient:
    """
    Thin wrapper around the Langfuse SDK.

    Design principles:
    - Every public method is wrapped in try/except — Langfuse NEVER breaks trading
    - All flushing is non-blocking (SDK daemon thread)
    - Prompt cache is handled by the SDK (configurable TTL)
    """

    def __init__(self):
        from app.config import get_settings

        self._settings = get_settings()
        self._lf = None
        self._enabled = False
        self._initialize()

    def _initialize(self) -> None:
        if not self._settings.langfuse_configured or not self._settings.langfuse_enabled:
            logger.info(
                "Langfuse disabled",
                reason=(
                    "not configured"
                    if not self._settings.langfuse_configured
                    else "LANGFUSE_ENABLED=false"
                ),
            )
            return

        try:
            from langfuse import Langfuse

            self._lf = Langfuse(
                public_key=self._settings.langfuse_public_key,
                secret_key=self._settings.langfuse_secret_key,
                host=self._settings.langfuse_host,
            )
            self._enabled = True
            logger.info("Langfuse initialized", host=self._settings.langfuse_host)
        except Exception as e:
            logger.warning(
                "Langfuse initialization failed — observability disabled",
                error=str(e),
            )
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled and self._lf is not None

    def trace(
        self,
        name: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Create a root trace. Returns a Langfuse Trace or _NoOpTrace.
        """
        if not self.enabled:
            return _NOOP
        try:
            return self._lf.trace(
                name=name,
                session_id=session_id,
                user_id=user_id,
                tags=tags or [],
                metadata=metadata or {},
            )
        except Exception as e:
            logger.warning("Failed to create Langfuse trace", error=str(e))
            return _NOOP

    def get_prompt(
        self,
        prompt_name: str,
        fallback: Optional[str] = None,
        cache_ttl_seconds: int = 300,
    ):
        """
        Fetch a prompt from Langfuse Prompt Management.

        Returns:
            (langfuse_prompt_object, compiled_text)
            langfuse_prompt_object is needed to link generations to prompt versions.
            Returns (None, fallback) on any failure.
        """
        if not self.enabled:
            return None, fallback
        try:
            prompt = self._lf.get_prompt(prompt_name, cache_ttl_seconds=cache_ttl_seconds)
            return prompt, prompt.prompt
        except Exception as e:
            logger.warning(
                "Failed to fetch prompt from Langfuse — using local fallback",
                prompt_name=prompt_name,
                error=str(e),
            )
            return None, fallback

    def flush(self) -> None:
        """Flush pending events to Langfuse. Call on shutdown."""
        if not self.enabled:
            return
        try:
            self._lf.flush()
        except Exception:
            pass

    def shutdown(self) -> None:
        """Flush and shutdown. Call from FastAPI lifespan."""
        if not self.enabled:
            return
        try:
            self._lf.shutdown()
        except Exception:
            pass


@lru_cache(maxsize=1)
def get_langfuse() -> LangfuseClient:
    """Get the singleton Langfuse client."""
    return LangfuseClient()

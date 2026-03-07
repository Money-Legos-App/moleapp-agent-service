"""
Langfuse Observability Client — Singleton (SDK v3)

Uses the Langfuse Python SDK v3 (OpenTelemetry-based).
All methods are no-ops when Langfuse is disabled or misconfigured.
Langfuse NEVER raises — every public method is wrapped in try/except.
"""

from functools import lru_cache
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


class _NoOpObservation:
    """Returned when Langfuse is disabled. All methods are safe no-ops."""

    def update(self, **kwargs) -> "_NoOpObservation":
        return self

    def update_trace(self, **kwargs) -> "_NoOpObservation":
        return self

    def end(self, **kwargs) -> "_NoOpObservation":
        return self

    def start_observation(self, **kwargs) -> "_NoOpObservation":
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# Singleton no-op instance
_NOOP = _NoOpObservation()


class LangfuseClient:
    """
    Thin wrapper around the Langfuse SDK v3.

    Design principles:
    - Every public method is wrapped in try/except — Langfuse NEVER breaks trading
    - All flushing is non-blocking (SDK background thread)
    - Prompt cache is handled by the SDK (configurable TTL)

    SDK v3 key concepts:
    - get_client() returns a singleton client
    - start_as_current_observation() creates spans/generations as context managers
    - start_observation() creates spans with manual lifecycle (.end())
    - propagate_attributes() sets session_id, user_id, tags on all children
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
            import os
            from langfuse import get_client

            # SDK v3 reads env vars automatically. Ensure they're set from our settings.
            os.environ.setdefault("LANGFUSE_PUBLIC_KEY", self._settings.langfuse_public_key)
            os.environ.setdefault("LANGFUSE_SECRET_KEY", self._settings.langfuse_secret_key)
            os.environ.setdefault("LANGFUSE_BASE_URL", self._settings.langfuse_host)

            self._lf = get_client()
            self._enabled = True
            logger.info("Langfuse initialized (SDK v3)", host=self._settings.langfuse_host)
        except Exception as e:
            logger.warning(
                "Langfuse initialization failed — observability disabled",
                error=str(e),
            )
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled and self._lf is not None

    def start_trace(
        self,
        name: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Create a root span (trace) with optional attributes.

        In SDK v3, traces are implicitly created by the first root span.
        We use start_observation() for manual lifecycle control (no context manager needed).

        Returns a Langfuse observation or _NoOpObservation.
        """
        if not self.enabled:
            return _NOOP
        try:
            from langfuse import propagate_attributes

            # Set trace-level attributes that propagate to all children
            attrs = {}
            if session_id:
                attrs["session_id"] = session_id
            if user_id:
                attrs["user_id"] = user_id
            if tags:
                attrs["tags"] = tags
            if metadata:
                attrs["metadata"] = metadata

            # propagate_attributes is a context manager, but we need manual control.
            # Instead, create a root span and update trace attributes on it.
            root = self._lf.start_observation(name=name, as_type="span")

            # Set trace-level attributes
            if attrs:
                root.update_trace(**attrs)

            return root
        except Exception as e:
            logger.warning("Failed to create Langfuse trace", error=str(e))
            return _NOOP

    def start_span(self, parent, name: str, **kwargs):
        """
        Create a child span under a parent observation.

        Returns a Langfuse observation or _NoOpObservation.
        """
        if not self.enabled or parent is _NOOP:
            return _NOOP
        try:
            return parent.start_observation(name=name, as_type="span", **kwargs)
        except Exception as e:
            logger.warning("Failed to create Langfuse span", error=str(e))
            return _NOOP

    def start_generation(self, parent, name: str, **kwargs):
        """
        Create a generation (LLM call) under a parent observation.

        Returns a Langfuse observation or _NoOpObservation.
        """
        if not self.enabled or parent is _NOOP:
            return _NOOP
        try:
            return parent.start_observation(name=name, as_type="generation", **kwargs)
        except Exception as e:
            logger.warning("Failed to create Langfuse generation", error=str(e))
            return _NOOP

    def log_event(self, parent, name: str, **kwargs):
        """
        Log an event under a parent observation.
        In v3, events are spans with zero duration.
        """
        if not self.enabled or parent is _NOOP:
            return _NOOP
        try:
            event = parent.start_observation(name=name, as_type="span")
            if kwargs:
                event.update(**kwargs)
            event.end()
            return event
        except Exception as e:
            logger.warning("Failed to log Langfuse event", error=str(e))
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
            (langfuse_prompt_object, raw_template_text)
            Returns (None, fallback) on any failure.

        Note: The returned prompt object has a .compile(**kwargs) method
        that handles {{variable}} substitution (Mustache syntax).
        Callers should use prompt.compile() for Langfuse prompts,
        NOT Python str.format() which uses {variable} syntax.
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

"""
Qwen LLM Client for Trading Analysis (LLM Competition)

Mirrors DeepSeekClient exactly — same interface, same prompts, same JSON format.
Only the API endpoint and model name differ. This ensures a fair competition.
"""

import json
import time as _time
from typing import Any, Dict, List, Optional

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from .prompts import PromptTemplates

logger = structlog.get_logger(__name__)


class QwenClient:
    """
    Client for Qwen (Alibaba Cloud) LLM API.

    Uses the OpenAI-compatible endpoint (DashScope).
    Same 3 methods as DeepSeekClient for drop-in competition:
    - analyze_market()
    - filter_for_user()
    - analyze_position_exit()
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        from app.config import get_settings

        settings = get_settings()
        self.api_key = api_key or settings.qwen_api_key
        self.api_url = api_url or settings.qwen_api_url
        self.model = model or settings.qwen_model

        if not self.api_key:
            logger.warning("Qwen API key not configured")

        self._client: Optional[httpx.AsyncClient] = None
        self._last_usage: Dict[str, Any] = {}
        self._last_raw_response: Optional[str] = None
        self._last_response_time_ms: Optional[int] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=60.0,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def _call_api(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1000,
        trace=None,
        lf_prompt=None,
        generation_name: str = "llm_call",
        generation_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        client = await self._get_client()

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }

        if temperature > 0.4:
            payload["top_p"] = 0.9

        logger.info(
            "AUDIT LLM request",
            model=self.model,
            provider="qwen",
            message_count=len(messages),
            prompt_chars=sum(len(m.get("content", "")) for m in messages),
            temperature=temperature,
        )

        # Langfuse generation span
        generation = None
        if trace is not None:
            try:
                from app.services.observability.langfuse_client import get_langfuse
                lf = get_langfuse()
                generation = lf.start_generation(trace, name=generation_name)
                gen_update = dict(
                    model=self.model,
                    input=messages,
                    metadata=generation_metadata or {},
                )
                gen_update["metadata"]["temperature"] = temperature
                gen_update["metadata"]["max_tokens"] = max_tokens
                gen_update["metadata"]["provider"] = "qwen"
                if lf_prompt is not None:
                    gen_update["metadata"]["prompt_name"] = getattr(lf_prompt, "name", str(lf_prompt))
                generation.update(**gen_update)
            except Exception:
                generation = None

        _api_start = _time.monotonic()

        response = await client.post(
            f"{self.api_url}/chat/completions",
            json=payload,
        )
        response.raise_for_status()

        self._last_response_time_ms = int((_time.monotonic() - _api_start) * 1000)

        data = response.json()
        content = data["choices"][0]["message"]["content"]

        self._last_usage = data.get("usage", {})
        self._last_raw_response = content

        logger.info(
            "AUDIT LLM response",
            model=self.model,
            provider="qwen",
            tokens_prompt=self._last_usage.get("prompt_tokens"),
            tokens_completion=self._last_usage.get("completion_tokens"),
            tokens_total=self._last_usage.get("total_tokens"),
            response_chars=len(content),
            response_time_ms=self._last_response_time_ms,
        )

        # Finalize Langfuse generation
        if generation is not None:
            try:
                from app.config import get_settings
                settings = get_settings()
                prompt_tokens = self._last_usage.get("prompt_tokens", 0)
                completion_tokens = self._last_usage.get("completion_tokens", 0)

                generation.update(
                    output=content,
                    usage_details={
                        "input": prompt_tokens,
                        "output": completion_tokens,
                        "total": self._last_usage.get("total_tokens", 0),
                    },
                    metadata={
                        "input_cost": prompt_tokens * settings.qwen_cost_per_input_token,
                        "output_cost": completion_tokens * settings.qwen_cost_per_output_token,
                        "total_cost": (
                            prompt_tokens * settings.qwen_cost_per_input_token
                            + completion_tokens * settings.qwen_cost_per_output_token
                        ),
                        "provider": "qwen",
                        "response_time_ms": self._last_response_time_ms,
                    },
                )
                generation.end()
            except Exception:
                pass

        return content

    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
                return json.loads(json_str)
            raise

    async def analyze_market(
        self,
        asset: str,
        current_price: float,
        price_change_24h: float,
        volume_24h: float,
        spread: float,
        pattern_context: Optional[str] = None,
        risk_metrics: Optional[Dict[str, Any]] = None,
        funding_rate: float = 0.0,
        open_interest: float = 0.0,
        tf_summary: Optional[str] = None,
        oi_delta: Optional[Dict[str, Any]] = None,
        bid_imbalance_pct: float = 0.0,
        trace=None,
        lf_prompt=None,
    ) -> Dict[str, Any]:
        """Analyze market conditions — identical interface to DeepSeekClient."""
        user_prompt = PromptTemplates.format_market_analysis(
            asset=asset,
            current_price=current_price,
            price_change_24h=price_change_24h,
            volume_24h=volume_24h,
            spread=spread,
            pattern_context=pattern_context or "",
            risk_metrics=risk_metrics or {},
            funding_rate=funding_rate,
            open_interest=open_interest,
            tf_summary=tf_summary,
            oi_delta=oi_delta,
            bid_imbalance_pct=bid_imbalance_pct,
        )

        messages = [
            {"role": "system", "content": PromptTemplates.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = await self._call_api(
                messages,
                temperature=0.55,
                trace=trace,
                lf_prompt=lf_prompt,
                generation_name="market_analysis",
                generation_metadata={"asset": asset, "provider": "qwen"},
            )
            signal = self._parse_json_response(response)

            signal["asset"] = asset
            signal["generated_at"] = None

            signal["_audit_prompt"] = user_prompt
            signal["_audit_response"] = response
            signal["_audit_model"] = self.model
            signal["_audit_tokens"] = self._last_usage.get("total_tokens")
            signal["_response_time_ms"] = self._last_response_time_ms

            logger.info(
                "Market analysis completed",
                asset=asset,
                provider="qwen",
                should_trade=signal.get("should_trade"),
                direction=signal.get("direction"),
                confidence=signal.get("confidence"),
            )

            return signal

        except Exception as e:
            logger.error("Market analysis failed", asset=asset, provider="qwen", error=str(e))
            return {
                "should_trade": False,
                "direction": None,
                "confidence": "LOW",
                "reasoning": f"Analysis failed: {str(e)}",
                "asset": asset,
                "_audit_prompt": user_prompt,
                "_audit_response": None,
                "_audit_model": self.model,
                "_audit_tokens": None,
                "_response_time_ms": None,
                "_audit_error": str(e),
            }

    async def filter_for_user(
        self,
        signal: Dict[str, Any],
        mission: Dict[str, Any],
        existing_positions: List[Dict[str, Any]],
        margin_used: float = 0.0,
        account_value: float = 0.0,
        trace=None,
        lf_prompt=None,
    ) -> Dict[str, Any]:
        """User-specific trade filter — identical interface to DeepSeekClient."""
        user_prompt = PromptTemplates.format_user_filter(
            signal=signal,
            mission=mission,
            existing_positions=existing_positions,
            margin_used=margin_used,
            account_value=account_value,
        )

        messages = [
            {"role": "system", "content": PromptTemplates.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = await self._call_api(
                messages,
                temperature=0.2,
                trace=trace,
                lf_prompt=lf_prompt,
                generation_name="user_filter",
                generation_metadata={"mission_id": mission.get("id"), "asset": signal.get("asset"), "provider": "qwen"},
            )
            decision = self._parse_json_response(response)

            decision["_audit_prompt"] = user_prompt
            decision["_audit_response"] = response
            decision["_audit_model"] = self.model
            decision["_audit_tokens"] = self._last_usage.get("total_tokens")
            decision["_response_time_ms"] = self._last_response_time_ms

            logger.info(
                "User filter completed",
                mission_id=mission.get("id"),
                provider="qwen",
                should_execute=decision.get("should_execute"),
            )

            return decision

        except Exception as e:
            logger.error("User filter failed", mission_id=mission.get("id"), provider="qwen", error=str(e))
            return {
                "should_execute": False,
                "skip_reason": f"Filter failed: {str(e)}",
                "_audit_prompt": user_prompt,
                "_audit_response": None,
                "_audit_model": self.model,
                "_audit_tokens": None,
                "_response_time_ms": None,
                "_audit_error": str(e),
            }

    async def analyze_position_exit(
        self,
        position: Dict[str, Any],
        mission: Dict[str, Any],
        market_data: Dict[str, Any],
        trace=None,
        lf_prompt=None,
    ) -> Dict[str, Any]:
        """Position exit analysis — identical interface to DeepSeekClient."""
        user_prompt = PromptTemplates.format_position_exit(
            position=position,
            mission=mission,
            market_data=market_data,
        )

        messages = [
            {"role": "system", "content": PromptTemplates.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = await self._call_api(
                messages,
                temperature=0.2,
                trace=trace,
                lf_prompt=lf_prompt,
                generation_name="position_exit",
                generation_metadata={"asset": position.get("asset"), "provider": "qwen"},
            )
            decision = self._parse_json_response(response)

            decision["_audit_prompt"] = user_prompt
            decision["_audit_response"] = response
            decision["_audit_model"] = self.model
            decision["_audit_tokens"] = self._last_usage.get("total_tokens")
            decision["_response_time_ms"] = self._last_response_time_ms

            logger.info(
                "Position exit analysis completed",
                position_id=position.get("id"),
                provider="qwen",
                should_exit=decision.get("should_exit"),
            )

            return decision

        except Exception as e:
            logger.error("Position exit analysis failed", position_id=position.get("id"), provider="qwen", error=str(e))
            return {
                "should_exit": False,
                "urgency": "hold",
                "reasoning": f"Analysis failed: {str(e)}",
                "_audit_prompt": user_prompt,
                "_audit_response": None,
                "_audit_model": self.model,
                "_audit_tokens": None,
                "_response_time_ms": None,
                "_audit_error": str(e),
            }

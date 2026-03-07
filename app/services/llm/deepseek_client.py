"""
DeepSeek LLM Client for Trading Analysis
"""

import json
from typing import Any, Dict, List, Optional

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from .prompts import PromptTemplates

logger = structlog.get_logger(__name__)


class DeepSeekClient:
    """
    Client for DeepSeek LLM API.

    Used for:
    - Market analysis and signal generation
    - User-specific trade filtering
    - Position exit decisions
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Initialize the DeepSeek client."""
        from app.config import get_settings

        settings = get_settings()
        self.api_key = api_key or settings.deepseek_api_key
        self.api_url = api_url or settings.deepseek_api_url
        self.model = model or settings.deepseek_model

        if not self.api_key:
            logger.warning("DeepSeek API key not configured")

        self._client: Optional[httpx.AsyncClient] = None
        # Audit: stores last API call metadata for callers to read
        self._last_usage: Dict[str, Any] = {}
        self._last_raw_response: Optional[str] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
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
        """Close the HTTP client."""
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
        # Langfuse instrumentation (all optional — defaults to no-op)
        trace=None,
        lf_prompt=None,
        generation_name: str = "llm_call",
        generation_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Make a request to the DeepSeek API.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (lower = more deterministic)
            max_tokens: Maximum response tokens
            trace: Langfuse trace object for observability
            lf_prompt: Langfuse prompt object to link generation to prompt version
            generation_name: Name for the Langfuse generation span
            generation_metadata: Extra metadata for the generation

        Returns:
            Response content string
        """
        client = await self._get_client()

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }

        logger.info(
            "AUDIT LLM request",
            model=self.model,
            message_count=len(messages),
            prompt_chars=sum(len(m.get("content", "")) for m in messages),
            temperature=temperature,
        )

        # Start Langfuse generation BEFORE the API call
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
                if lf_prompt is not None:
                    gen_update["metadata"]["prompt_name"] = getattr(lf_prompt, "name", str(lf_prompt))
                generation.update(**gen_update)
            except Exception:
                generation = None

        response = await client.post(
            f"{self.api_url}/chat/completions",
            json=payload,
        )
        response.raise_for_status()

        data = response.json()
        content = data["choices"][0]["message"]["content"]

        # Store usage metadata for audit logging
        self._last_usage = data.get("usage", {})
        self._last_raw_response = content

        logger.info(
            "AUDIT LLM response",
            model=self.model,
            tokens_prompt=self._last_usage.get("prompt_tokens"),
            tokens_completion=self._last_usage.get("completion_tokens"),
            tokens_total=self._last_usage.get("total_tokens"),
            response_chars=len(content),
        )

        # Finalize Langfuse generation with usage and cost
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
                        "input_cost": prompt_tokens * settings.deepseek_cost_per_input_token,
                        "output_cost": completion_tokens * settings.deepseek_cost_per_output_token,
                        "total_cost": (
                            prompt_tokens * settings.deepseek_cost_per_input_token
                            + completion_tokens * settings.deepseek_cost_per_output_token
                        ),
                    },
                )
                generation.end()
            except Exception:
                pass

        return content

    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        try:
            # Try direct parsing
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code block
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
        # Langfuse instrumentation
        trace=None,
        lf_prompt=None,
    ) -> Dict[str, Any]:
        """
        Analyze market conditions and generate a trading signal.

        Args:
            asset: Asset symbol (e.g., "ETH-USD")
            current_price: Current market price
            price_change_24h: 24-hour price change percentage
            volume_24h: 24-hour trading volume
            spread: Bid-ask spread percentage
            pattern_context: Context from similar historical patterns (None if RAG disabled)
            risk_metrics: Risk metrics from FAISS patterns (None if RAG disabled)
            funding_rate: Current funding rate (hourly, raw)
            open_interest: Current open interest in USD
            tf_summary: Multi-timeframe technical analysis string
            oi_delta: OI change and volume delta vs previous cycle
            bid_imbalance_pct: Orderbook bid/ask imbalance percentage

        Returns:
            Trading signal dictionary
        """
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
                temperature=0.3,
                trace=trace,
                lf_prompt=lf_prompt,
                generation_name="market_analysis",
                generation_metadata={"asset": asset},
            )
            signal = self._parse_json_response(response)

            # Add metadata
            signal["asset"] = asset
            signal["generated_at"] = None  # Will be set by caller

            # Attach audit data for workflow nodes to persist
            signal["_audit_prompt"] = user_prompt
            signal["_audit_response"] = response
            signal["_audit_model"] = self.model
            signal["_audit_tokens"] = self._last_usage.get("total_tokens")

            logger.info(
                "Market analysis completed",
                asset=asset,
                should_trade=signal.get("should_trade"),
                direction=signal.get("direction"),
                confidence=signal.get("confidence"),
            )

            return signal

        except Exception as e:
            logger.error(
                "Market analysis failed",
                asset=asset,
                error=str(e),
            )
            # Return safe default with audit data
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
                "_audit_error": str(e),
            }

    async def filter_for_user(
        self,
        signal: Dict[str, Any],
        mission: Dict[str, Any],
        existing_positions: List[Dict[str, Any]],
        # Langfuse instrumentation
        trace=None,
        lf_prompt=None,
    ) -> Dict[str, Any]:
        """
        Determine if a signal should be executed for a specific user.

        Args:
            signal: Trading signal from market analysis
            mission: User's mission parameters
            existing_positions: User's current open positions

        Returns:
            Execution decision dictionary
        """
        user_prompt = PromptTemplates.format_user_filter(
            signal=signal,
            mission=mission,
            existing_positions=existing_positions,
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
                generation_metadata={"mission_id": mission.get("id"), "asset": signal.get("asset")},
            )
            decision = self._parse_json_response(response)

            # Attach audit data
            decision["_audit_prompt"] = user_prompt
            decision["_audit_response"] = response
            decision["_audit_model"] = self.model
            decision["_audit_tokens"] = self._last_usage.get("total_tokens")

            logger.info(
                "User filter completed",
                mission_id=mission.get("id"),
                should_execute=decision.get("should_execute"),
            )

            return decision

        except Exception as e:
            logger.error(
                "User filter failed",
                mission_id=mission.get("id"),
                error=str(e),
            )
            return {
                "should_execute": False,
                "skip_reason": f"Filter failed: {str(e)}",
                "_audit_prompt": user_prompt,
                "_audit_response": None,
                "_audit_model": self.model,
                "_audit_tokens": None,
                "_audit_error": str(e),
            }

    async def analyze_position_exit(
        self,
        position: Dict[str, Any],
        mission: Dict[str, Any],
        market_data: Dict[str, Any],
        # Langfuse instrumentation
        trace=None,
        lf_prompt=None,
    ) -> Dict[str, Any]:
        """
        Analyze whether a position should be exited.

        Args:
            position: Current position details
            mission: Mission parameters
            market_data: Current market conditions

        Returns:
            Exit decision dictionary
        """
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
                generation_metadata={"asset": position.get("asset"), "direction": position.get("direction")},
            )
            decision = self._parse_json_response(response)

            # Attach audit data
            decision["_audit_prompt"] = user_prompt
            decision["_audit_response"] = response
            decision["_audit_model"] = self.model
            decision["_audit_tokens"] = self._last_usage.get("total_tokens")

            logger.info(
                "Position exit analysis completed",
                position_id=position.get("id"),
                should_exit=decision.get("should_exit"),
                reason=decision.get("exit_reason"),
            )

            return decision

        except Exception as e:
            logger.error(
                "Position exit analysis failed",
                position_id=position.get("id"),
                error=str(e),
            )
            # Default to holding position on error
            return {
                "should_exit": False,
                "urgency": "hold",
                "reasoning": f"Analysis failed: {str(e)}",
                "_audit_prompt": user_prompt,
                "_audit_response": None,
                "_audit_model": self.model,
                "_audit_tokens": None,
                "_audit_error": str(e),
            }

    async def batch_analyze_markets(
        self,
        assets: List[str],
        market_data: Dict[str, Dict[str, Any]],
        pattern_contexts: Dict[str, str],
        risk_metrics: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple assets in batch.

        Args:
            assets: List of asset symbols
            market_data: Market data keyed by asset
            pattern_contexts: Pattern contexts keyed by asset
            risk_metrics: Risk metrics keyed by asset

        Returns:
            List of trading signals
        """
        import asyncio

        tasks = []
        for asset in assets:
            data = market_data.get(asset, {})
            tasks.append(
                self.analyze_market(
                    asset=asset,
                    current_price=data.get("price", 0),
                    price_change_24h=data.get("price_change_24h", 0),
                    volume_24h=data.get("volume_24h", 0),
                    spread=data.get("spread", 0.01),
                    pattern_context=pattern_contexts.get(asset, "No historical context"),
                    risk_metrics=risk_metrics.get(asset, {}),
                    funding_rate=data.get("funding_rate", 0),
                    open_interest=data.get("open_interest", 0),
                )
            )

        signals = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_signals = []
        for signal in signals:
            if isinstance(signal, Exception):
                logger.error("Batch analysis error", error=str(signal))
                continue
            valid_signals.append(signal)

        return valid_signals

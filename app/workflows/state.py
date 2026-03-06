"""
LangGraph State Definitions
Defines the state schema for the trading workflow
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict


class StrategySignal(TypedDict):
    """Trading signal from market analysis."""

    asset: str
    direction: str  # "LONG" or "SHORT"
    confidence: str  # "LOW", "MEDIUM", "HIGH"
    recommended_leverage: int
    strategy_tag: str
    reasoning: str
    entry_zone: Dict[str, float]
    stop_loss_percent: float
    take_profit_percent: float
    time_horizon: str
    rag_context_ids: List[str]
    max_drawdown_30d: Optional[float]
    volatility_score: Optional[float]
    generated_at: datetime


class ExecutionPayload(TypedDict):
    """Payload for trade execution."""

    mission_id: str
    user_id: str
    action: str  # "ENTER_LONG", "ENTER_SHORT", "EXIT_LONG", "EXIT_SHORT", "FORCE_CLOSE"
    asset: str
    quantity: float
    price: Optional[float]
    leverage: int
    stop_loss_price: Optional[float]
    take_profit_price: Optional[float]
    signal_id: Optional[str]
    decision_reason: str


class ExecutionResult(TypedDict):
    """Result of trade execution."""

    mission_id: str
    success: bool
    order_id: Optional[str]
    tx_hash: Optional[str]
    error: Optional[str]
    executed_price: Optional[float]
    executed_quantity: Optional[float]


class AgentState(TypedDict):
    """
    State schema for the LangGraph trading workflow.

    This state is passed between nodes as the workflow executes.
    """

    # Trigger info
    trigger_type: str  # "scheduled", "price_alert", "manual"
    triggered_at: datetime

    # Layer A: Market Analysis outputs
    signals: List[StrategySignal]
    rag_context_ids: List[str]
    market_data: Dict[str, Dict[str, Any]]  # asset -> market data
    pattern_contexts: Dict[str, str]  # asset -> context string
    risk_metrics: Dict[str, Dict[str, Any]]  # asset -> risk metrics

    # Layer B: User Filter outputs
    eligible_missions: List[Dict[str, Any]]
    execution_payloads: List[ExecutionPayload]
    skip_reasons: Dict[str, str]  # mission_id -> reason skipped

    # Layer C: Execution outputs
    executed_orders: List[ExecutionResult]
    failed_orders: List[ExecutionResult]

    # Position monitoring outputs
    positions_to_close: List[Dict[str, Any]]
    position_alerts: List[Dict[str, Any]]

    # Workflow metadata
    errors: List[str]
    completed_nodes: List[str]

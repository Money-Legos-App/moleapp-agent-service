"""
LangGraph Trading Workflow
Orchestrates the three-layer trading agent workflow
"""

from datetime import datetime
from typing import Any, Dict, Literal

import structlog
from langgraph.graph import END, StateGraph

from app.workflows.state import AgentState
from app.workflows.nodes import (
    market_analysis_node,
    user_filter_node,
    execution_node,
    position_monitoring_node,
)

logger = structlog.get_logger(__name__)


def should_continue_to_execution(state: AgentState) -> Literal["execution", "end"]:
    """
    Determine if we should proceed to execution.

    Continue if we have execution payloads or positions to close.
    """
    has_payloads = len(state.get("execution_payloads", [])) > 0
    has_positions_to_close = len(state.get("positions_to_close", [])) > 0

    if has_payloads or has_positions_to_close:
        return "execution"
    return "end"


def should_continue_to_user_filter(state: AgentState) -> Literal["user_filter", "end"]:
    """
    Determine if we should proceed to user filter.

    Continue if we have signals to process.
    """
    has_signals = len(state.get("signals", [])) > 0
    if has_signals:
        return "user_filter"
    return "end"


class TradingWorkflow:
    """
    LangGraph-based trading workflow.

    Layers:
    - Layer A (market_analysis): Fetches data, queries FAISS, generates signals via DeepSeek
    - Layer B (user_filter): Filters signals for eligible missions
    - Layer C (execution): Signs and submits orders via wallet-service

    Additional nodes:
    - position_monitoring: Monitors open positions for risk events
    """

    def __init__(self):
        """Initialize the workflow graph."""
        self._graph = None
        self._compiled = None

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph."""
        # Create the graph with our state schema
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("market_analysis", market_analysis_node)
        workflow.add_node("user_filter", user_filter_node)
        workflow.add_node("execution", execution_node)
        workflow.add_node("position_monitoring", position_monitoring_node)

        # Define the flow
        # Start with market analysis
        workflow.set_entry_point("market_analysis")

        # After market analysis, check if we have signals
        workflow.add_conditional_edges(
            "market_analysis",
            should_continue_to_user_filter,
            {
                "user_filter": "user_filter",
                "end": END,
            },
        )

        # After user filter, check if we have execution payloads
        workflow.add_conditional_edges(
            "user_filter",
            should_continue_to_execution,
            {
                "execution": "execution",
                "end": END,
            },
        )

        # After execution, run position monitoring
        workflow.add_edge("execution", "position_monitoring")

        # Position monitoring ends the workflow
        workflow.add_edge("position_monitoring", END)

        return workflow

    def compile(self):
        """Compile the workflow graph."""
        if self._compiled is None:
            self._graph = self._build_graph()
            self._compiled = self._graph.compile()
        return self._compiled

    async def run(
        self,
        trigger_type: str = "scheduled",
        initial_state: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Run the trading workflow.

        Args:
            trigger_type: What triggered this run ("scheduled", "price_alert", "manual")
            initial_state: Optional initial state overrides

        Returns:
            Final workflow state
        """
        compiled = self.compile()

        # Build initial state
        state: AgentState = {
            "trigger_type": trigger_type,
            "triggered_at": datetime.utcnow(),
            "signals": [],
            "rag_context_ids": [],
            "market_data": {},
            "pattern_contexts": {},
            "risk_metrics": {},
            "eligible_missions": [],
            "execution_payloads": [],
            "skip_reasons": {},
            "executed_orders": [],
            "failed_orders": [],
            "positions_to_close": [],
            "position_alerts": [],
            "errors": [],
            "completed_nodes": [],
        }

        # Apply any initial state overrides
        if initial_state:
            state.update(initial_state)

        logger.info(
            "Starting trading workflow",
            trigger_type=trigger_type,
        )

        try:
            # Run the workflow
            final_state = await compiled.ainvoke(state)

            logger.info(
                "Trading workflow completed",
                completed_nodes=final_state.get("completed_nodes", []),
                signals_generated=len(final_state.get("signals", [])),
                orders_executed=len(final_state.get("executed_orders", [])),
                orders_failed=len(final_state.get("failed_orders", [])),
                errors=len(final_state.get("errors", [])),
            )

            return final_state

        except Exception as e:
            logger.error(
                "Trading workflow failed",
                error=str(e),
                exc_info=True,
            )
            raise


# Module-level function for easy access
async def run_trading_workflow(
    trigger_type: str = "scheduled",
    initial_state: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Run the trading workflow.

    This is the main entry point called by the scheduler.

    Args:
        trigger_type: What triggered this run
        initial_state: Optional state overrides

    Returns:
        Final workflow state
    """
    workflow = TradingWorkflow()
    return await workflow.run(trigger_type=trigger_type, initial_state=initial_state)


# Alternative workflow for position monitoring only
class PositionMonitoringWorkflow:
    """
    Simplified workflow for position monitoring only.

    Used by the 5-minute monitoring scheduler.
    """

    def __init__(self):
        self._graph = None
        self._compiled = None

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        workflow.add_node("position_monitoring", position_monitoring_node)
        workflow.set_entry_point("position_monitoring")
        workflow.add_edge("position_monitoring", END)
        return workflow

    def compile(self):
        if self._compiled is None:
            self._graph = self._build_graph()
            self._compiled = self._graph.compile()
        return self._compiled

    async def run(self) -> Dict[str, Any]:
        compiled = self.compile()

        state: AgentState = {
            "trigger_type": "monitoring",
            "triggered_at": datetime.utcnow(),
            "signals": [],
            "rag_context_ids": [],
            "market_data": {},
            "pattern_contexts": {},
            "risk_metrics": {},
            "eligible_missions": [],
            "execution_payloads": [],
            "skip_reasons": {},
            "executed_orders": [],
            "failed_orders": [],
            "positions_to_close": [],
            "position_alerts": [],
            "errors": [],
            "completed_nodes": [],
        }

        return await compiled.ainvoke(state)


async def run_position_monitoring() -> Dict[str, Any]:
    """Run position monitoring workflow."""
    workflow = PositionMonitoringWorkflow()
    return await workflow.run()

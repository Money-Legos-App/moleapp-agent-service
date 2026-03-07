"""
Playbook Model

Structured output from the Slow Thinker (LLM), consumed by the Fast Actor
(deterministic execution engine) for sub-second entries/exits.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


class PlaybookStatus(str, Enum):
    PENDING = "PENDING"        # Waiting for price to enter entry zone
    ENTERED = "ENTERED"        # Position opened, monitoring exits
    EXPIRED = "EXPIRED"        # TTL elapsed without entry
    COMPLETED = "COMPLETED"    # Exited (TP/SL/trailing)
    CANCELLED = "CANCELLED"    # Cancelled by new cycle or risk


@dataclass
class Playbook:
    # Identity
    playbook_id: str
    cycle_id: str
    mission_id: str
    asset: str                  # "ETH-USD"

    # Direction & Sizing
    direction: str              # "LONG" or "SHORT"
    position_size: float        # asset units (pre-calculated for this mission)
    leverage: int
    margin_allocated: float     # USDC margin

    # Entry Conditions
    entry_price: float          # ideal entry
    entry_zone_min: float       # lower bound
    entry_zone_max: float       # upper bound
    max_slippage_pct: float

    # Exit Conditions
    stop_loss_price: float
    take_profit_price: float
    trailing_activation_pct: float  # % profit before trailing activates
    trailing_callback_pct: float    # % pullback from peak to trigger close

    # Metadata
    conviction: str             # "HIGH", "MEDIUM", "LOW"
    strategy_tag: str
    reasoning: str
    ttl_seconds: int = 900      # 15 min = 1 cycle

    # Timestamps
    created_at: float = field(default_factory=time.time)

    # Mutable state (updated by Fast Actor)
    status: PlaybookStatus = PlaybookStatus.PENDING
    entered_at: Optional[float] = None
    entered_price: Optional[float] = None
    position_id: Optional[str] = None

    @property
    def is_expired(self) -> bool:
        return time.time() > self.created_at + self.ttl_seconds

    @property
    def redis_key(self) -> str:
        return f"agent:playbook:{self.playbook_id}"

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)

    @classmethod
    def from_json(cls, raw: str) -> "Playbook":
        data = json.loads(raw)
        data["status"] = PlaybookStatus(data["status"])
        return cls(**data)


# Redis key constants
PLAYBOOKS_PENDING_KEY = "agent:playbooks:pending"
PLAYBOOKS_ACTIVE_KEY = "agent:playbooks:active"

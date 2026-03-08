"""
Alkebulan Agent Service Configuration
Pydantic Settings for environment-based configuration
"""

from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Service info
    service_name: str = "alkebulan-agent"
    service_version: str = "1.0.0"
    environment: str = Field(default="development", alias="NODE_ENV")
    debug: bool = False

    # Server
    host: str = "0.0.0.0"
    port: int = 3006

    # Database (shared with other services)
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/wallet_backend_unified",
        alias="DATABASE_URL",
    )

    # Wallet Service Bridge (for Turnkey signing)
    wallet_service_url: str = Field(
        default="http://wallet-service:3002",
        alias="WALLET_SERVICE_URL",
    )
    wallet_service_api_key: str = Field(
        default="",
        alias="WALLET_SERVICE_API_KEY",
    )

    # User Service (for mission state)
    user_service_url: str = Field(
        default="http://user-service:3001",
        alias="USER_SERVICE_URL",
    )

    # Keycloak Configuration (for service-to-service auth)
    keycloak_url: str = Field(
        default="http://keycloak:8080",
        alias="KEYCLOAK_URL",
    )
    keycloak_realm: str = Field(
        default="moleapp",
        alias="KEYCLOAK_REALM",
    )
    keycloak_client_id: str = Field(
        default="agent-service",
        alias="KEYCLOAK_CLIENT_ID",
    )
    keycloak_client_secret: str = Field(
        default="",
        alias="KEYCLOAK_CLIENT_SECRET",
    )

    # DeepSeek LLM
    deepseek_api_key: str = Field(default="", alias="DEEPSEEK_API_KEY")
    deepseek_api_url: str = Field(
        default="https://api.deepseek.com/v1",
        alias="DEEPSEEK_API_URL",
    )
    deepseek_model: str = Field(
        default="deepseek-chat",
        alias="DEEPSEEK_MODEL",
    )

    # Hyperliquid
    hyperliquid_api_url: str = Field(
        default="https://api.hyperliquid-testnet.xyz",
        alias="HYPERLIQUID_API_URL",
    )
    hyperliquid_mainnet: bool = Field(default=False, alias="HYPERLIQUID_MAINNET")
    moleapp_agent_address: str = Field(
        default="",
        alias="MOLEAPP_AGENT_ADDRESS",
        description="MoleApp's backend agent wallet address for Hyperliquid",
    )

    # Redis (shared with other services)
    redis_url: str = Field(
        default="redis://redis-cache:6379",
        alias="REDIS_URL",
    )
    redis_password: Optional[str] = Field(
        default=None,
        alias="REDIS_PASSWORD",
    )

    # Execution Fleet
    execution_concurrency: int = Field(
        default=10,
        alias="EXECUTION_CONCURRENCY",
        description="Max concurrent trade execution workers",
    )
    hl_rate_limit_per_second: float = Field(
        default=5.0,
        alias="HL_RATE_LIMIT_PER_SECOND",
        description="Legacy: max HL API calls per second (superseded by weight-based limiter)",
    )
    execution_job_timeout_seconds: int = Field(
        default=30,
        alias="EXECUTION_JOB_TIMEOUT_SECONDS",
        description="Max time for a single trade execution job",
    )

    # Hyperliquid Rate Limiting (weight-based)
    hl_weight_budget_per_minute: float = Field(
        default=1100.0,
        alias="HL_WEIGHT_BUDGET_PER_MINUTE",
        description="Global weight budget per minute (HL limit: 1200, leave 100 headroom)",
    )
    hl_info_weight_budget: float = Field(
        default=900.0,
        alias="HL_INFO_WEIGHT_BUDGET",
        description="Info endpoint weight budget per minute",
    )
    hl_exchange_weight_budget: float = Field(
        default=200.0,
        alias="HL_EXCHANGE_WEIGHT_BUDGET",
        description="Exchange endpoint weight budget per minute (reserved for trading)",
    )

    # WebSocket Market Data Feed
    ws_enabled: bool = Field(
        default=True,
        alias="WS_ENABLED",
        description="Enable WebSocket feed for real-time market data",
    )
    ws_reconnect_max_delay: int = Field(
        default=60,
        alias="WS_RECONNECT_MAX_DELAY",
        description="Max reconnect delay in seconds for WS",
    )

    # Fast Actor (deterministic execution engine)
    fast_actor_enabled: bool = Field(
        default=False,
        alias="FAST_ACTOR_ENABLED",
        description="Enable Fast Actor for sub-second playbook execution",
    )
    fast_actor_max_slippage_pct: float = Field(
        default=0.5,
        alias="FAST_ACTOR_MAX_SLIPPAGE_PCT",
        description="Max slippage % for Fast Actor entries",
    )
    playbook_ttl_seconds: int = Field(
        default=900,
        alias="PLAYBOOK_TTL_SECONDS",
        description="How long a playbook stays valid for entry (default: 15 min)",
    )
    fast_actor_use_ioc_exits: bool = Field(
        default=True,
        alias="FAST_ACTOR_USE_IOC_EXITS",
        description="Use IOC orders for exits (immediate fill)",
    )
    fast_actor_entry_band_pct: float = Field(
        default=0.3,
        alias="FAST_ACTOR_ENTRY_BAND_PCT",
        description="Entry zone width % around ideal price",
    )

    # Dynamic Asset Rotation
    dynamic_asset_rotation_enabled: bool = Field(
        default=False,
        alias="DYNAMIC_ASSET_ROTATION_ENABLED",
        description="Enable daily volume-based asset rotation",
    )
    dynamic_asset_rotation_hour: int = Field(
        default=1,
        alias="DYNAMIC_ASSET_ROTATION_HOUR",
        description="UTC hour to run rotation (1 = after daily close settles)",
    )
    dynamic_asset_min_volume_24h: float = Field(
        default=15_000_000,
        alias="DYNAMIC_ASSET_MIN_VOLUME_24H",
        description="Minimum 24h volume ($) to qualify for rotation",
    )
    dynamic_asset_max_count: int = Field(
        default=15,
        alias="DYNAMIC_ASSET_MAX_COUNT",
        description="Max assets in the rotated list",
    )

    # PnL Fan-Out
    pnl_fanout_batch_size: int = Field(
        default=5,
        alias="PNL_FANOUT_BATCH_SIZE",
        description="Number of missions to enqueue per batch during PnL fan-out",
    )
    pnl_fanout_jitter_max_ms: int = Field(
        default=2000,
        alias="PNL_FANOUT_JITTER_MAX_MS",
        description="Max random jitter (ms) per PnL job to prevent thundering herd",
    )

    # arq Worker Queue
    arq_max_jobs: int = Field(
        default=10,
        alias="ARQ_MAX_JOBS",
        description="Max concurrent arq worker jobs",
    )
    arq_job_timeout_seconds: int = Field(
        default=30,
        alias="ARQ_JOB_TIMEOUT_SECONDS",
    )
    deposit_check_max_attempts: int = Field(
        default=10,
        alias="DEPOSIT_CHECK_MAX_ATTEMPTS",
        description="Max deposit check retries before giving up (~30 min)",
    )
    deposit_check_initial_delay_seconds: int = Field(
        default=120,
        alias="DEPOSIT_CHECK_INITIAL_DELAY_SECONDS",
    )
    market_price_cache_ttl_seconds: int = Field(
        default=90,
        alias="MARKET_PRICE_CACHE_TTL_SECONDS",
        description="Max age of cached market prices before stale",
    )
    # FAISS Vector Store
    disable_rag: bool = Field(
        default=False,
        alias="DISABLE_RAG",
        description="Skip FAISS/RAG initialization to save memory (512MB Render plans)",
    )
    faiss_index_path: str = Field(
        default="/app/data/faiss_index",
        alias="FAISS_INDEX_PATH",
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )

    # Scheduler
    analysis_interval_minutes: int = Field(
        default=15,
        alias="ANALYSIS_INTERVAL_MINUTES",
        description="How often to run market analysis (Layer A)",
    )
    pnl_sync_interval_seconds: int = Field(
        default=60,
        alias="PNL_SYNC_INTERVAL_SECONDS",
        description="How often to sync positions from Hyperliquid",
    )
    pnl_snapshot_hour: int = Field(
        default=0,
        alias="PNL_SNAPSHOT_HOUR",
        description="Hour (UTC) to take daily PnL snapshots",
    )

    # Trading constraints
    # Note: stored as comma-separated string to avoid pydantic-settings JSON parsing
    allowed_assets_str: str = Field(
        default="BTC-USD,ETH-USD,SOL-USD,SUI-USD,SEI-USD,AVAX-USD,TAO-USD,FET-USD,NEAR-USD,WIF-USD,POPCAT-USD,kPEPE-USD,DOGE-USD,PENDLE-USD,ARB-USD",
        alias="ALLOWED_ASSETS",
    )
    max_leverage_conservative: int = 1
    max_leverage_moderate: int = 2
    max_leverage_aggressive: int = 3

    # Mission defaults
    default_mission_duration_days: int = 30
    min_deposit_usdc: float = 10.0
    max_deposit_usdc: float = 10000.0

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: str = "json"

    # Metrics
    metrics_enabled: bool = True
    metrics_port: int = 9090

    # DRY RUN Mode (Paper Trading)
    dry_run: bool = Field(
        default=True,
        alias="DRY_RUN",
        description="If True, orders will be logged but not executed on Hyperliquid",
    )
    dry_run_log_trades: bool = Field(
        default=True,
        alias="DRY_RUN_LOG_TRADES",
        description="If True, log full trade details in dry run mode",
    )

    # HL Treasury (testnet auto-funding via usdSend)
    hl_treasury_private_key: str = Field(
        default="",
        alias="HL_TREASURY_PRIVATE_KEY",
        description="Private key for the HL testnet treasury wallet (auto-funds users on activation)",
    )
    hl_treasury_address: str = Field(
        default="",
        alias="HL_TREASURY_ADDRESS",
        description="Address of the HL testnet treasury wallet",
    )

    # HCP Vault Cloud (Transit Engine for per-mission Master EOA keys)
    hcp_vault_url: str = Field(
        default="",
        alias="HCP_VAULT_URL",
        description="HCP Vault cluster URL (e.g., https://<cluster>.vault.<region>.hashicorp.cloud:8200)",
    )
    hcp_vault_token: str = Field(
        default="",
        alias="HCP_VAULT_TOKEN",
        description="HCP Vault service principal token (legacy — prefer AppRole)",
    )
    hcp_vault_namespace: str = Field(
        default="admin",
        alias="HCP_VAULT_NAMESPACE",
        description="HCP Vault namespace",
    )
    vault_transit_mount: str = Field(
        default="transit",
        alias="VAULT_TRANSIT_MOUNT",
        description="Vault Transit secrets engine mount point",
    )
    # Vault AppRole auth (preferred over static token)
    vault_role_id: str = Field(
        default="",
        alias="VAULT_ROLE_ID",
        description="Vault AppRole role ID",
    )
    vault_secret_id: str = Field(
        default="",
        alias="VAULT_SECRET_ID",
        description="Vault AppRole secret ID",
    )

    # Platform Treasury (Arbitrum — collects 30% profit fee)
    platform_treasury_address: str = Field(
        default="",
        alias="PLATFORM_TREASURY_ADDRESS",
        description="Platform treasury wallet address on Arbitrum for fee collection",
    )
    profit_fee_percent: float = Field(
        default=30.0,
        alias="PROFIT_FEE_PERCENT",
        description="Percentage of profit taken as platform fee (only on positive yield)",
    )

    # Gas Station (funds Master EOAs with ETH for Arbitrum TXs)
    gas_station_private_key: str = Field(
        default="",
        alias="GAS_STATION_PRIVATE_KEY",
        description="Private key for the Gas Station wallet on Arbitrum",
    )
    gas_station_address: str = Field(
        default="",
        alias="GAS_STATION_ADDRESS",
        description="Gas Station wallet address on Arbitrum",
    )

    # Langfuse Observability + Prompt Management
    langfuse_public_key: str = Field(default="", alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(default="", alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com",
        alias="LANGFUSE_HOST",
    )
    langfuse_enabled: bool = Field(
        default=True,
        alias="LANGFUSE_ENABLED",
        description="Master switch for Langfuse. Set False in tests or if keys not configured.",
    )
    deepseek_cost_per_input_token: float = Field(
        default=0.00000014,
        alias="DEEPSEEK_COST_PER_INPUT_TOKEN",
        description="DeepSeek cost per input token in USD ($0.14/1M)",
    )
    deepseek_cost_per_output_token: float = Field(
        default=0.00000028,
        alias="DEEPSEEK_COST_PER_OUTPUT_TOKEN",
        description="DeepSeek cost per output token in USD ($0.28/1M)",
    )

    # Risk Enforcement
    risk_enforcement_enabled: bool = Field(
        default=True,
        alias="RISK_ENFORCEMENT_ENABLED",
        description="Master switch for deterministic risk enforcement (SL/TP/trailing/drawdown)",
    )
    risk_liquidation_protection_percent: float = Field(
        default=5.0,
        alias="RISK_LIQUIDATION_PROTECTION_PERCENT",
        description="Close positions when within this % of liquidation price",
    )

    # Dynamic Stop-Loss Scaling
    dynamic_sl_enabled: bool = Field(
        default=True,
        alias="DYNAMIC_SL_ENABLED",
        description="Enable leverage-aware dynamic stop-loss calculation",
    )
    dynamic_sl_buffer_pct: float = Field(
        default=20.0,
        alias="DYNAMIC_SL_BUFFER_PCT",
        description="Liquidation buffer % for dynamic SL (higher = wider SL). "
        "At 1-3x leverage, profile SL wins; dynamic SL is a safety net for higher leverage.",
    )

    # Minimum Notional (Hyperliquid rejects orders below ~$10 notional)
    hl_min_notional_usd: float = Field(
        default=10.0,
        alias="HL_MIN_NOTIONAL_USD",
        description="Minimum order notional in USD. HL rejects orders below this. "
        "If a position is too small, the engine consolidates max_positions to 1 for that cycle.",
    )

    # Minimum Reward-to-Risk Ratio (hard kill switch — never trust LLM on R/R)
    min_reward_risk_ratio: float = Field(
        default=2.0,
        alias="MIN_REWARD_RISK_RATIO",
        description="Minimum TP%/SL% ratio. Orders with R/R below this are killed. "
        "Protects against LLM recommending bad risk/reward setups.",
    )

    # Slippage-Aware Position Sizing
    slippage_sizing_enabled: bool = Field(
        default=True,
        alias="SLIPPAGE_SIZING_ENABLED",
        description="Enable market-stress-based position size reduction",
    )
    slippage_max_reduction_pct: float = Field(
        default=30.0,
        alias="SLIPPAGE_MAX_REDUCTION_PCT",
        description="Maximum position size reduction under market stress (%)",
    )

    # Correlation Bucketing
    correlation_bucketing_enabled: bool = Field(
        default=True,
        alias="CORRELATION_BUCKETING_ENABLED",
        description="Enable beta-bucket exposure caps to prevent correlated concentration",
    )
    correlation_bucket_btc_cap: float = Field(
        default=50.0,
        alias="CORRELATION_BUCKET_BTC_CAP",
        description="Max % of total leverage budget for BTC-correlated assets",
    )
    correlation_bucket_eth_cap: float = Field(
        default=35.0,
        alias="CORRELATION_BUCKET_ETH_CAP",
        description="Max % of total leverage budget for ETH-correlated assets",
    )
    correlation_bucket_uncorrelated_cap: float = Field(
        default=15.0,
        alias="CORRELATION_BUCKET_UNCORRELATED_CAP",
        description="Max % of total leverage budget for uncorrelated assets",
    )

    # Infrastructure Resilience (Heartbeat + Backup Monitor)
    heartbeat_ttl_seconds: int = Field(
        default=600,
        alias="HEARTBEAT_TTL_SECONDS",
        description="Risk monitor heartbeat TTL (should be > 2x monitoring interval)",
    )
    heartbeat_block_new_entries: bool = Field(
        default=True,
        alias="HEARTBEAT_BLOCK_NEW_ENTRIES",
        description="Block new position entries when risk monitor heartbeat is missing",
    )

    # Circuit Breaker
    circuit_breaker_failure_threshold: int = Field(
        default=5,
        alias="CIRCUIT_BREAKER_FAILURE_THRESHOLD",
        description="Number of failures before circuit trips",
    )
    circuit_breaker_window_minutes: int = Field(
        default=30,
        alias="CIRCUIT_BREAKER_WINDOW_MINUTES",
        description="Time window for tracking failures",
    )
    circuit_breaker_reset_minutes: int = Field(
        default=60,
        alias="CIRCUIT_BREAKER_RESET_MINUTES",
        description="How long to wait before resetting a tripped circuit",
    )

    @property
    def allowed_assets(self) -> List[str]:
        """Parse allowed assets from comma-separated string."""
        return [asset.strip() for asset in self.allowed_assets_str.split(",") if asset.strip()]

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_testnet(self) -> bool:
        return not self.hyperliquid_mainnet

    @property
    def langfuse_configured(self) -> bool:
        """True if Langfuse API keys are set."""
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @property
    def vault_configured(self) -> bool:
        """True if Vault URL + at least one auth method (AppRole or token) are set."""
        if not self.hcp_vault_url:
            return False
        return bool(self.vault_role_id and self.vault_secret_id) or bool(self.hcp_vault_token)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Hyperliquid chain configuration
class HyperliquidConfig:
    """Hyperliquid-specific configuration."""

    TESTNET_API = "https://api.hyperliquid-testnet.xyz"
    MAINNET_API = "https://api.hyperliquid.xyz"
    TESTNET_WS = "wss://api.hyperliquid-testnet.xyz/ws"
    MAINNET_WS = "wss://api.hyperliquid.xyz/ws"

    # Agent approval uses SDK Exchange.approve_agent() — no contract addresses needed

    # Asset mappings
    ASSET_DECIMALS = {
        "BTC-USD": 8,
        "ETH-USD": 18,
        "SOL-USD": 9,
        "SUI-USD": 9,
        "SEI-USD": 18,
        "AVAX-USD": 18,
        "TAO-USD": 18,
        "FET-USD": 18,
        "NEAR-USD": 18,
        "WIF-USD": 6,
        "POPCAT-USD": 18,
        "kPEPE-USD": 18,
        "DOGE-USD": 8,
        "PENDLE-USD": 18,
        "ARB-USD": 18,
    }

    @classmethod
    def get_api_url(cls, is_mainnet: bool) -> str:
        return cls.MAINNET_API if is_mainnet else cls.TESTNET_API

    @classmethod
    def get_ws_url(cls, is_mainnet: bool) -> str:
        return cls.MAINNET_WS if is_mainnet else cls.TESTNET_WS

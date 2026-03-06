# Alkebulan Agent Service

AI-powered DeFi trading agent for MoleApp. Handles market analysis, signal generation, and trade execution on Hyperliquid.

## Architecture

- **Layer A**: Market Analysis (RAG + DeepSeek LLM)
- **Layer B**: User Filter (mission validation, risk checks)
- **Layer C**: Execution (Turnkey signing via wallet-service, Hyperliquid submission)

## Key Features

- LangGraph workflow orchestration
- FAISS vector store for historical pattern matching
- Circuit breaker for failure protection
- DRY_RUN mode for paper trading

## Configuration

See `app/config/settings.py` for environment variables.

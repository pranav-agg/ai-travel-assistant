"""Central configuration. Import this rather than reading os.environ directly."""


from __future__ import annotations


import os
from pathlib import Path
from zoneinfo import ZoneInfo


from dotenv import load_dotenv


load_dotenv()


# --- Paths --------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
RAW_DIR = KNOWLEDGE_BASE_DIR / "raw"
SOURCES_FILE = KNOWLEDGE_BASE_DIR / "sources.json"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"
CACHE_DIR = PROJECT_ROOT / ".cache"
MCP_SERVERS_DIR = PROJECT_ROOT / "mcp_servers"


# --- Models -------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
EMBEDDING_REQUEST_INTERVAL_SECONDS = float(
    os.getenv("EMBEDDING_REQUEST_INTERVAL_SECONDS", "0")
)


# --- Vector store -------------------------------------------------------
COLLECTION_NAME = "singapore_travel"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RETRIEVAL_K = 5
RETRIEVAL_FETCH_K = 20
RETRIEVAL_LAMBDA = 0.6
RELEVANCE_THRESHOLD = 0.25


# --- Destination --------------------------------------------------------.
DESTINATION = "Singapore"
DESTINATION_TZ = ZoneInfo("Asia/Singapore")
DESTINATION_CURRENCY = "SGD"


CITY_COORDS: dict[str, tuple[float, float]] = {
    "singapore": (1.3521, 103.8198),
    "johor bahru": (1.4927, 103.7414),
    "batam": (1.0456, 104.0305),
    "kuala lumpur": (3.1390, 101.6869),
}


# --- External services (both keyless) -----------------------------------
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
FRANKFURTER_BASE = "https://api.frankfurter.dev/v2"
HTTP_TIMEOUT_SECONDS = 10.0


# Open-Meteo forecasts at most 16 days ahead.
FORECAST_HORIZON_DAYS = 16


REQUIRED_CURRENCIES = frozenset({"INR", "USD", "SGD"})
# Frozen allowlist so a hiccup on the currency-list endpoint degrades
# validation rather than breaking conversions mid-demo.
FALLBACK_CURRENCIES = frozenset(
    {
        "AUD", "BGN", "BRL", "CAD", "CHF", "CNY", "CZK", "DKK", "EUR", "GBP",
        "HKD", "HUF", "IDR", "ILS", "INR", "ISK", "JPY", "KRW", "MXN", "MYR",
        "NOK", "NZD", "PHP", "PLN", "RON", "SEK", "SGD", "THB", "TRY", "USD",
        "ZAR",
    }
)


# --- Demo aids ----------------------------------------------------------
# Set to "weather" or "currency" to force that server to return a structured
# error, so the failure path can be demonstrated (PLAN.md §6 failure handling).
DEMO_FORCE_TOOL_FAILURE = os.getenv("DEMO_FORCE_TOOL_FAILURE", "").strip().lower()

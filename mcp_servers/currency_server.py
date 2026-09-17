"""MCP server: currency conversion for travel budgeting.

Backed by Frankfurter (https://frankfurter.dev) — keyless, no quota.

Frankfurter serves ECB *reference rates*, published once per working day
at ~16:00 CET, and not at all on weekends or Eurosystem holidays.


There are no real-time ticks to miss. A Saturday request does not fail and
does not return a gap — it silently returns Friday's rate. The risk is
therefore presenting a two-day-old reference rate as a live quote, so every
response carries `rate_date` and a `staleness` label, and the assistant is

forbidden from saying "live" or "current" unless it genuinely is today's.


Two further properties worth disclosing to the user:
  * ECB rates are EUR-based, so INR->SGD is triangulated through EUR.
  * The ECB states these are "for information purposes only" — not the rate
    anyone actually transacts at.


Run standalone:  python mcp_servers/currency_server.py
"""


from __future__ import annotations


import json
import sys
from datetime import date, datetime
from pathlib import Path


import httpx
from mcp.server.fastmcp import FastMCP


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from src.config import (  # noqa: E402
    CACHE_DIR,
    DEMO_FORCE_TOOL_FAILURE,
    DESTINATION_TZ,
    FALLBACK_CURRENCIES,
    FRANKFURTER_BASE,
    HTTP_TIMEOUT_SECONDS,
    REQUIRED_CURRENCIES,
)
from src.dates import classify_staleness, staleness_note, today_local  # noqa: E402


mcp = FastMCP("currency")


CACHE_FILE = CACHE_DIR / "fx_rates.json"


DISCLAIMER = (
    "ECB reference rate, published once per working day and EUR-based "
    "(cross-rates are triangulated through EUR). "
)


# Starts as the frozen allowlist and is upgraded to the live list on first
# tool use. Deliberately NOT populated at startup -- see
# `_refresh_supported_currencies` for why that would stop the server booting.
_supported: frozenset[str] = FALLBACK_CURRENCIES
_checked: bool = False




def _error(message: str, *, recoverable: bool = True, **extra) -> dict:
    return {"error": message, "tool": "convert_currency",
            "recoverable": recoverable, **extra}




def _refresh_supported_currencies() -> None:
    """Replace the frozen allowlist with the live list, if it can be fetched.


    Called lazily on first tool use -- NEVER at import or startup. An MCP
    server communicates over stdio and must begin serving immediately; the
    client gives it a short window to complete the handshake. Doing network
    I/O before `mcp.run()` means that on a slow, proxied or firewalled
    network the server is still waiting on an HTTP timeout when that window
    closes, so it never connects at all. That failure looks identical to a
    crash, which makes it genuinely hard to diagnose.


    Failure here is never fatal either. The frozen allowlist already covers
    every currency this app needs, so an unreachable list endpoint degrades
    *validation quality* and nothing else. A conversion that truly cannot be
    served fails later, at call time, with a structured error.
    """
    global _supported, _checked
    _checked = True


    try:
        with httpx.Client(timeout=HTTP_TIMEOUT_SECONDS) as client:
            response = client.get(f"{FRANKFURTER_BASE}/currencies")
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        print(
            f"[currency] Could not fetch the live currency list ({exc}); "
            f"using the built-in allowlist of {len(_supported)} codes.",
            file=sys.stderr,
        )
        return


    live = frozenset(str(code).upper() for code in payload)
    if not live:
        return


    missing = REQUIRED_CURRENCIES - live
    if missing:
        # Report it, but keep serving: the frozen allowlist still has these.
        print(
            f"[currency] WARNING: {sorted(missing)} absent from the upstream "
            f"list; keeping the built-in allowlist.",
            file=sys.stderr,
        )
        return


    _supported = live
    print(
        f"[currency] Verified {len(_supported)} currencies; "
        f"{', '.join(sorted(REQUIRED_CURRENCIES))} all supported.",
        file=sys.stderr,
    )




def _ensure_currencies_loaded() -> None:
    """Refresh the currency list once per process, on first use."""
    if not _checked:
        _refresh_supported_currencies()




def _read_cache(pair: str) -> dict | None:
    try:
        data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        return data.get(pair)
    except (OSError, ValueError):
        return None




def _write_cache(pair: str, rate: float, rate_date: str) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        data[pair] = {"rate": rate, "rate_date": rate_date}
        CACHE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass  # A non-writable cache must never break a conversion.




def _build_response(
    amount: float,
    base: str,
    quote: str,
    rate: float,
    rate_date: date,
    today: date,
    *,
    from_cache: bool = False,
) -> dict:
    underlying = classify_staleness(rate_date)
    staleness = "cached" if from_cache else underlying
    note = (
        "Served from a locally cached rate because the rate service was "
        f"unreachable. {staleness_note(underlying, rate_date)}"
        if from_cache
        else staleness_note(underlying, rate_date)
    )
    return {
        "source": "Frankfurter (European Central Bank reference rates)",
        "source_url": "https://frankfurter.dev/",
        "retrieved_at": datetime.now(DESTINATION_TZ).isoformat(timespec="seconds"),
        "amount": round(amount, 2),
        "from_currency": base,
        "to_currency": quote,
        "rate": rate,
        "converted": round(amount * rate, 2),
        "rate_date": rate_date.isoformat(),
        "rate_date_weekday": rate_date.strftime("%A"),
        "age_days": (today - rate_date).days,
        "staleness": staleness,
        "staleness_note": note,
        "basis": "EUR-based ECB reference rate"
        + (" (cross-rate via EUR)" if "EUR" not in (base, quote) else ""),
        "disclaimer": DISCLAIMER,
    }




@mcp.tool()
def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert an amount between two currencies.


    Uses ECB reference rates, which are published once per working day. On a
    weekend or holiday the most recent working day's rate is returned - the
    response's `staleness` and `staleness_note` say which. Quote the rate
    date to the user; never describe the result as a live or current rate
    unless `staleness` is "current".


    Args:
        amount: How much to convert. Must be positive.
        from_currency: ISO 4217 code, e.g. "INR".
        to_currency: ISO 4217 code, e.g. "SGD".
    """
    if DEMO_FORCE_TOOL_FAILURE == "currency":
        return _error(
            "Currency service unavailable (DEMO_FORCE_TOOL_FAILURE is set). "
            "Tell the user the conversion could not be performed and do not "
            "estimate a rate."
        )


    base = from_currency.strip().upper()
    quote = to_currency.strip().upper()
    today = today_local(DESTINATION_TZ)


    if amount <= 0:
        return _error(f"Amount must be positive, got {amount}.")


    _ensure_currencies_loaded()


    for code in (base, quote):
        if len(code) != 3 or not code.isalpha():
            return _error(f"'{code}' is not a valid ISO 4217 currency code.")
        if code not in _supported:
            return _error(
                f"{code} is not available from this rate source. "
                f"Supported codes include: "
                f"{', '.join(sorted(REQUIRED_CURRENCIES))}.",
                supported_sample=sorted(_supported)[:40],
            )


    # Same-currency conversions need no network call and no rate date.
    if base == quote:
        return {
            "source": "n/a (same currency)",
            "amount": round(amount, 2),
            "from_currency": base,
            "to_currency": quote,
            "rate": 1.0,
            "converted": round(amount, 2),
            "staleness": "current",
            "staleness_note": "Same currency - no conversion applied.",
        }


    pair = f"{base}{quote}"
    url = f"{FRANKFURTER_BASE}/rate/{base.lower()}/{quote.lower()}"


    last_exc: Exception | None = None
    for _ in range(2):  # one retry, then fall back to cache
        try:
            with httpx.Client(timeout=HTTP_TIMEOUT_SECONDS) as client:
                response = client.get(url)
                response.raise_for_status()
                payload = response.json()


            # Frankfurter v2 single-pair responses are flat:
            # {"date": "...", "base": "INR", "quote": "SGD", "rate": ...}
            # Older/v1-style responses used {"rates": {"SGD": ...}}, so keep
            # that parser as a defensive fallback.
            rate = payload.get("rate")
            if rate is None:
                rates = payload.get("rates") or {}
                rate = rates.get(quote) or rates.get(quote.upper())
            raw_date = payload.get("date")
            if rate is None or raw_date is None:
                return _error(
                    f"Rate source returned no rate for {base}->{quote}."
                )


            rate_date = date.fromisoformat(raw_date)
            _write_cache(pair, float(rate), raw_date)
            return _build_response(
                amount, base, quote, float(rate), rate_date, today
            )
        except (httpx.HTTPError, ValueError) as exc:
            last_exc = exc


    cached = _read_cache(pair)
    if cached:
        return _build_response(
            amount,
            base,
            quote,
            float(cached["rate"]),
            date.fromisoformat(cached["rate_date"]),
            today,
            from_cache=True,
        )


    return _error(
        f"Could not retrieve an exchange rate for {base}->{quote} "
        f"({last_exc}). No cached rate is available. Do not estimate a rate; "
        "tell the user the conversion is currently unavailable."
    )




@mcp.tool()
def list_supported_currencies() -> dict:
    """List the ISO 4217 currency codes this rate source supports."""
    _ensure_currencies_loaded()
    return {
        "source": "Frankfurter (European Central Bank reference rates)",
        "count": len(_supported),
        "required_codes_available": sorted(REQUIRED_CURRENCIES & _supported),
        "codes": sorted(_supported),
        "note": (
            "Rates are published once per working day (~16:00 CET) and not "
            "on weekends or Eurosystem holidays."
        ),
    }




if __name__ == "__main__":
    mcp.run(transport="stdio")

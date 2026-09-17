"""Date anchoring and FX rate staleness.

An LLM has no clock. if user asks "next week", so every
relative date phrase has to be resolved against a real calendar somewhere.
This module is that somewhere: it is the single source of truth for "today",
and everything else derives from it.

Two rules the rest of the codebase depends on:

1. `build_date_anchor()` is called **per invocation**, never cached. A
   module-level `TODAY = date.today()` goes stale the moment the Streamlit
   process outlives midnight, which is the normal case for a long-running app.
2. Dates that appear in a rendered itinerary are *copied* from tool payloads,
   never recomputed by the model. This module only helps it choose a window.

Deliberately dependency-free apart from the stdlib so both MCP servers can
import it without dragging in LangChain.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from src.config import FORECAST_HORIZON_DAYS

DEST_TZ = ZoneInfo("Asia/Singapore")

# The ECB publishes euro reference rates around 16:00 CET on working days.
# Staleness is judged on *this* clock, not the destination's: "is there a
# newer rate?" is a question about Frankfurt's publication schedule. Judging
# it in Singapore time (UTC+8, so 23:00 at publication) would label almost
# every genuinely fresh rate as a day old.
ECB_TZ = ZoneInfo("Europe/Berlin")
ECB_PUBLICATION_HOUR_CET = 16

# The widest gap a normal Eurosystem closure produces.
DELAYED_MAX_LAG_DAYS = 5


def today_local(tz: ZoneInfo = DEST_TZ) -> date:
    """Today in the *destination's* timezone.

    Not the user's. At 23:00 IST it is already tomorrow in Singapore, and the
    traveller's experience is what the itinerary describes.
    """
    return datetime.now(tz).date()


def next_monday(d: date) -> date:
    """The next Monday strictly after `d`.

    If `d` is itself a Monday this returns the Monday a week later, which is
    what a person means by "next Monday" when speaking on a Monday.
    """
    days_ahead = (7 - d.weekday()) % 7
    return d + timedelta(days=days_ahead or 7)


def upcoming_weekend(d: date) -> tuple[date, date]:
    """The (Saturday, Sunday) pair a person means by "this weekend".

    On a Sunday we are already inside the weekend, so the remaining weekend is
    today only — returning next Saturday would be wrong.
    """
    if d.weekday() == 6:  # Sunday
        return d, d
    saturday = d + timedelta(days=(5 - d.weekday()) % 7)
    return saturday, saturday + timedelta(days=1)


def forecast_horizon(d: date) -> date:
    """Last date Open-Meteo can forecast from `d`."""
    return d + timedelta(days=FORECAST_HORIZON_DAYS)


def _fmt(d: date) -> str:
    return f"{d.isoformat()} ({d.strftime('%A')})"


def build_date_anchor(tz: ZoneInfo = DEST_TZ) -> str:
    """The date anchor block injected into the system prompt every turn.

    Five precomputed lines rather than a bare timestamp. The weekday names
    matter as much as the dates: "next Monday" requires knowing what day today
    *is*, and making the model derive weekday-from-ISO-date reintroduces
    exactly the arithmetic this is meant to remove.
    """
    today = today_local(tz)
    sat, sun = upcoming_weekend(today)
    weekend = _fmt(sat) if sat == sun else f"{sat.isoformat()} to {sun.isoformat()}"

    return (
        f"Today: {_fmt(today)}, {tz.key}\n"
        f"Tomorrow: {_fmt(today + timedelta(days=1))}\n"
        f"Next Monday: {next_monday(today).isoformat()}\n"
        f"Upcoming weekend: {weekend}\n"
        f"Forecast horizon: through {forecast_horizon(today).isoformat()} "
        f"({FORECAST_HORIZON_DAYS}-day API limit)"
    )


# --- Forecast window validation (§6.3 layer 3) --------------------------


@dataclass(frozen=True)
class WindowError:
    """A rejected forecast window, with the valid range spelled out.

    Returned instead of forwarding a doomed request, so a model slip surfaces
    as a clear message rather than an empty forecast.
    """

    message: str
    valid_from: str
    valid_to: str

    def as_dict(self) -> dict:
        return {
            "error": self.message,
            "valid_range": {"from": self.valid_from, "to": self.valid_to},
            "recoverable": True,
        }


def validate_window(
    start: date, num_days: int, today: date
) -> WindowError | None:
    """Reject windows Open-Meteo cannot serve. Returns None when valid."""
    horizon = forecast_horizon(today)

    if start < today:
        return WindowError(
            f"Cannot forecast into the past: {start.isoformat()} is before "
            f"today ({today.isoformat()}).",
            today.isoformat(),
            horizon.isoformat(),
        )
    if start > horizon:
        return WindowError(
            f"Forecasts reach only {FORECAST_HORIZON_DAYS} days ahead. "
            f"{start.isoformat()} is beyond that. For a trip this far out, "
            f"use typical seasonal conditions instead of a forecast.",
            today.isoformat(),
            horizon.isoformat(),
        )
    if num_days < 1:
        return WindowError(
            f"num_days must be at least 1, got {num_days}.",
            today.isoformat(),
            horizon.isoformat(),
        )
    return None


def clamp_num_days(start: date, num_days: int, today: date) -> int:
    """Trim a window that starts inside the horizon but runs past its end."""
    horizon = forecast_horizon(today)
    available = (horizon - start).days + 1
    return max(1, min(num_days, available))


# --- FX rate staleness (§6.4) -------------------------------------------


def latest_expected_publication(now_cet: datetime | None = None) -> date:
    """The most recent date the ECB should have published a rate for.

    Before 16:00 CET today's rate is not out yet, so the newest available is
    the previous working day's. Weekends never publish, so walk back to
    Friday. Eurosystem holidays are not modelled — they surface as a
    "delayed" classification rather than being predicted.
    """
    now = now_cet or datetime.now(ECB_TZ)
    d = now.date()
    if now.hour < ECB_PUBLICATION_HOUR_CET:
        d -= timedelta(days=1)
    while d.weekday() >= 5:  # Saturday or Sunday
        d -= timedelta(days=1)
    return d


def classify_staleness(rate_date: date, now_cet: datetime | None = None) -> str:
    """Label an ECB reference rate against what *should* be available.

    Frankfurter serves ECB reference rates: one publication per working day
    around 16:00 CET, and none at all on weekends or Eurosystem holidays.
    ".

    Returns one of:
        "current" — the freshest rate the ECB has published; nothing newer
                    exists, even if the date is not today
        "delayed" — a few days behind, consistent with a holiday cluster
        "stale"   — further behind than any normal closure explains
    """
    expected = latest_expected_publication(now_cet)
    if rate_date >= expected:
        return "current"
    lag = (expected - rate_date).days
    return "delayed" if lag <= DELAYED_MAX_LAG_DAYS else "stale"


def staleness_note(
    staleness: str, rate_date: date, now_cet: datetime | None = None
) -> str:
    """Human-readable phrasing the assistant can quote verbatim.

    Deliberately never says "live" or "real-time", and only says "today"
    when the rate really was published today — see the prompt rule in
    src/prompts.py.
    """
    pretty = rate_date.strftime("%a %d %b %Y")
    published_today = rate_date == datetime.now(ECB_TZ).date() if now_cet is None \
        else rate_date == now_cet.date()

    if staleness == "current":
        if published_today:
            return f"ECB reference rate published today ({pretty})."
        return (
            f"Latest published ECB reference rate ({pretty}). Rates are "
            "published once per working day and not on weekends or "
            "Eurosystem holidays, so no newer rate exists right now."
        )
    if staleness == "delayed":
        return (
            f"ECB reference rate for {pretty}, a few days behind the usual "
            "schedule — most likely a Eurosystem holiday period. Indicative only."
        )
    return (
        f"ECB reference rate for {pretty}, further behind than a normal "
        "closure explains. Treat it as indicative only and re-check before "
        "relying on it."
    )

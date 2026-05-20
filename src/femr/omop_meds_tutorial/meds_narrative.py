"""Render a MEDS subject's pre-prediction-time history as a narrative string.

Two outputs, in order:
  1. A leading **demographics block** (sex, race, ethnicity, date of birth,
     age at prediction time) extracted from the structural Truveta codes
     (``Race//*``, ``Gender//*``, ``Ethnicity//*``, ``MEDS_BIRTH``, …).
  2. A **timeline block**: events grouped by calendar date, with raw med
     codes translated to descriptions via the FEMR ontology.

The demographics events are removed from the dated timeline so they don't
clutter it. ``build_narrative`` returns the full string plus counts; pieces
can also be assembled separately via ``extract_demographics`` and
``build_timeline``.
"""
from __future__ import annotations

import collections
import datetime as _dt
from typing import Iterable, Optional


# Structural Truveta prefixes — these are not OMOP/Athena concepts, they
# encode patient demographics. We surface them in a dedicated block.
_STRUCTURAL_PREFIXES: tuple[str, ...] = (
    "SexAssignedAtBirth//",
    "GenderIdentity//",
    "Gender//",
    "Race//",
    "Ethnicity//",
)
_DEMOGRAPHIC_PSEUDO_CODES: frozenset[str] = frozenset({"MEDS_BIRTH", "MEDS_DEATH"})


def _is_demographic_code(code: str) -> bool:
    if code in _DEMOGRAPHIC_PSEUDO_CODES:
        return True
    return any(code.startswith(p) for p in _STRUCTURAL_PREFIXES)


def _describe(code: str, ontology) -> str:
    """Translate a med code into a human-readable description.

    Falls back through several sources because Truveta data mixes
    OMOP/Athena concept codes with non-ontology structural codes
    (``Race//White``, ``Gender//Female``, ``MEDS_BIRTH``, …).
    """
    if code in _DEMOGRAPHIC_PSEUDO_CODES:
        return code.replace("MEDS_", "").lower()
    for pref in _STRUCTURAL_PREFIXES:
        if code.startswith(pref):
            attr, val = code.split("//", 1)
            return f"{attr.replace('_', ' ').lower()}: {val}"
    desc = ontology.get_description(code) if ontology is not None else None
    if desc:
        return desc
    return code


def _format_event_value(ev) -> str:
    """Return a short string for the event's value (numeric or text)."""
    parts: list[str] = []
    nv = getattr(ev, "numeric_value", None)
    if nv is not None:
        unit = getattr(ev, "unit", None)
        parts.append(f"{nv:g}{(' ' + unit) if unit else ''}")
    tv = getattr(ev, "text_value", None)
    if tv:
        parts.append(str(tv))
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Demographics
# ---------------------------------------------------------------------------

def extract_demographics(subject, *, prediction_time: Optional[_dt.datetime] = None) -> dict:
    """Pull sex/gender/race/ethnicity/DOB from the subject's structural codes.

    Returns a dict with keys: ``sex_assigned_at_birth``, ``gender``,
    ``gender_identity``, ``race``, ``ethnicity``, ``date_of_birth``,
    ``age_at_prediction_years`` (None when ``prediction_time`` not supplied
    or DOB missing). Multiple values for a single attribute are joined
    with ``" / "``.
    """
    fields: dict[str, list[str]] = {
        "sex_assigned_at_birth": [],
        "gender": [],
        "gender_identity": [],
        "race": [],
        "ethnicity": [],
    }
    dob: Optional[_dt.datetime] = None
    for ev in subject.events:
        c = ev.code
        if c == "MEDS_BIRTH":
            if dob is None and ev.time is not None:
                dob = ev.time
            continue
        if c.startswith("SexAssignedAtBirth//"):
            fields["sex_assigned_at_birth"].append(c.split("//", 1)[1])
        elif c.startswith("GenderIdentity//"):
            fields["gender_identity"].append(c.split("//", 1)[1])
        elif c.startswith("Gender//"):
            fields["gender"].append(c.split("//", 1)[1])
        elif c.startswith("Race//"):
            fields["race"].append(c.split("//", 1)[1])
        elif c.startswith("Ethnicity//"):
            fields["ethnicity"].append(c.split("//", 1)[1])

    out: dict = {
        k: " / ".join(sorted(set(v))) if v else None for k, v in fields.items()
    }
    out["date_of_birth"] = dob.date() if dob else None

    age_years: Optional[float] = None
    if dob is not None and prediction_time is not None:
        delta = prediction_time - dob
        age_years = round(delta.days / 365.25, 1)
    out["age_at_prediction_years"] = age_years
    return out


def format_demographics(demo: dict) -> str:
    """Render the demographics dict as a bullet list (one line per field)."""
    lines: list[str] = ["## Demographics"]
    label_map = [
        ("sex_assigned_at_birth", "Sex assigned at birth"),
        ("gender", "Gender"),
        ("gender_identity", "Gender identity"),
        ("race", "Race"),
        ("ethnicity", "Ethnicity"),
    ]
    for key, label in label_map:
        val = demo.get(key)
        if val:
            lines.append(f"  - {label}: {val}")
    if demo.get("date_of_birth"):
        line = f"  - Date of birth: {demo['date_of_birth']}"
        if demo.get("age_at_prediction_years") is not None:
            line += f"  (age {demo['age_at_prediction_years']} at prediction time)"
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------

def build_timeline(
    subject,
    *,
    ontology,
    prediction_time: _dt.datetime,
    max_events: Optional[int] = None,
) -> tuple[str, int, int]:
    """Group non-demographic events by date and render them.

    Returns ``(timeline_str, n_events_kept, n_dates_kept)``. Dedupes
    ``(date, code, value)`` triples. When ``max_events`` is supplied,
    keeps the most recent dates that fit under the budget (whole dates
    only).
    """
    by_date: dict[_dt.date, list[tuple[str, str]]] = collections.defaultdict(list)
    seen: set[tuple[_dt.date, str, str]] = set()
    total = 0
    for ev in subject.events:
        if ev.time is None or ev.time > prediction_time:
            continue
        if _is_demographic_code(ev.code):
            continue
        desc = _describe(ev.code, ontology)
        val = _format_event_value(ev)
        key = (ev.time.date(), ev.code, val)
        if key in seen:
            continue
        seen.add(key)
        by_date[ev.time.date()].append((desc, val))
        total += 1

    dates = sorted(by_date.keys())
    if max_events is not None and total > max_events:
        kept_dates: list[_dt.date] = []
        running = 0
        for d in reversed(dates):
            n = len(by_date[d])
            if running + n > max_events and kept_dates:
                break
            kept_dates.append(d)
            running += n
        dates = sorted(kept_dates)
        total = sum(len(by_date[d]) for d in dates)

    lines: list[str] = ["## Timeline"]
    if max_events is not None and len(by_date) > len(dates):
        lines.append(
            f"(showing the most recent {len(dates)} of {len(by_date)} dates; "
            "earlier history truncated)"
        )
    for d in dates:
        items = by_date[d]
        lines.append(f"[{d.isoformat()}] ({len(items)} event{'s' if len(items) != 1 else ''})")
        grouped: dict[str, list[str]] = collections.defaultdict(list)
        for desc, val in items:
            grouped[desc].append(val)
        for desc, vals in grouped.items():
            nonempty = [v for v in vals if v]
            if nonempty:
                if len(nonempty) == 1:
                    lines.append(f"  - {desc}: {nonempty[0]}")
                else:
                    lines.append(f"  - {desc} (x{len(vals)}): {'; '.join(nonempty)}")
            else:
                if len(vals) > 1:
                    lines.append(f"  - {desc} (x{len(vals)})")
                else:
                    lines.append(f"  - {desc}")
    return "\n".join(lines), total, len(dates)


def build_narrative(
    subject,
    *,
    ontology,
    prediction_time: _dt.datetime,
    max_events: Optional[int] = None,
) -> tuple[str, int, int]:
    """Render the full narrative: demographics block followed by timeline.

    Returns ``(narrative_str, n_events_kept, n_dates_kept)``.
    """
    demo = extract_demographics(subject, prediction_time=prediction_time)
    demo_str = format_demographics(demo)
    timeline_str, n_events, n_dates = build_timeline(
        subject, ontology=ontology,
        prediction_time=prediction_time, max_events=max_events,
    )
    full = demo_str + "\n\n" + timeline_str
    return full, n_events, n_dates

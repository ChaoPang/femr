"""Use Claude (Anthropic API) to explain MOTOR predictions.

For one or more subjects, this script assembles two pieces of context:

  1. The MOTOR prediction header (true_label, predicted_prob, outcome_time)
  2. The patient's pre-prediction-time event trajectory, translated from
     raw med codes to human-readable concept descriptions via the FEMR
     ontology and grouped by date (with a leading demographics block)

Claude is asked to reason over the *narrative alone* and explain why the
MOTOR predicted probability is or isn't supported by the patient's
documented history. The MOTOR reasoning-layer tokens are intentionally
NOT shown to Claude — we want an independent clinical read of the
trajectory, not a post-hoc justification of MOTOR's attribution.

Usage:
    # 1) Validate the narrative + prompt (no Claude call) ----
    python -m femr.omop_meds_tutorial.explain_predictions \\
        --pretraining_data /path/to/motor \\
        --reasoning_parquet  /path/to/reasoning_predictions_long.parquet \\
        --meds_reader        /path/to/meds_reader \\
        --subject_id 4036872 \\
        --narrative_only

    # 2) When narrative looks right, run with Claude ----
    python -m femr.omop_meds_tutorial.explain_predictions \\
        --pretraining_data /path/to/motor \\
        --reasoning_parquet  /path/to/reasoning_predictions_long.parquet \\
        --meds_reader        /path/to/meds_reader \\
        --subject_id 4036872 \\
        --output explanations/4036872.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import pickle
import sys
import textwrap
from typing import Optional

import meds_reader
import polars as pl

from femr.omop_meds_tutorial.meds_narrative import build_narrative


# ---------------------------------------------------------------------------
# Per-subject metadata
# ---------------------------------------------------------------------------

def load_subject_meta(parquet_path: pathlib.Path, subject_id: int) -> dict:
    """Pull the per-subject prediction metadata row from the parquet."""
    df = pl.read_parquet(parquet_path).filter(pl.col("subject_id") == subject_id)
    if df.height == 0:
        raise SystemExit(f"subject_id={subject_id} not found in {parquet_path}")
    # Same metadata is replicated across rank rows; take rank=1.
    return df.filter(pl.col("rank") == 1).row(0, named=True)


# ---------------------------------------------------------------------------
# Claude prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a clinical informaticist evaluating a machine-learning prediction
    on a single patient. The model is MOTOR, a transformer-based foundation
    model trained on electronic health records to predict the probability
    that a patient with knee osteoarthritis (KOA) will undergo total knee
    replacement (TKR) within 5 years of their first KOA diagnosis.

    You will see only:
      (1) the MOTOR predicted probability for this patient, and
      (2) the patient's demographics and chronological event timeline up
          through the first KOA diagnosis.

    You will NOT be told the ground-truth outcome, the actual TKR date, or
    MOTOR's internal attribution. Reason about the prediction yourself,
    purely from the trajectory.

    Cohort context:
      - The base rate of 5-year TKR in this KOA cohort is ~17%.
      - Subjects with prior partial or total knee replacement, or any
        "history of TKR" code, were already excluded.
      - A 60-day washout drops TKRs coded within 60 days of the KOA
        diagnosis (same-encounter coding rather than incident outcome).

    Be concrete, cite specific dated events from the trajectory, and avoid
    boilerplate. Keep the explanation under 350 words.
""")


def build_user_prompt(
    *,
    subject_id: int,
    predicted_prob: Optional[float],
    prediction_time: _dt.datetime,
    narrative: str,
) -> str:
    sections = [
        f"## Patient {subject_id}",
        f"- prediction time (first KOA): {prediction_time.date().isoformat()}",
    ]
    if predicted_prob is not None:
        sections.append(f"- MOTOR predicted P(TKR within 5y): {predicted_prob:.4f}")
    else:
        sections.append("- MOTOR predicted P(TKR): N/A (subject not in test set)")

    sections += [
        "",
        "## Patient trajectory up to the prediction time",
        "Demographics first, then events grouped by calendar date.",
        "```",
        narrative,
        "```",
        "",
        "## Task",
        "Reason over the trajectory and assess MOTOR's prediction. Specifically:",
        "1. Summarize the clinical picture from the trajectory in 1-2 sentences.",
        "2. Identify evidence in the trajectory that supports a high or low",
        "   5-year TKR risk — cite specific dated events, not generic categories.",
        "3. State whether MOTOR's predicted probability is well-supported by the",
        "   timeline, and call out anything missing from the record that would",
        "   meaningfully change your assessment.",
    ]
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pretraining_data", required=True,
                   help="motor/ folder (for ontology.pkl)")
    p.add_argument("--reasoning_parquet", required=True,
                   help="Long-format reasoning predictions parquet "
                        "(only used as a metadata source: label, MOTOR p, outcome time)")
    p.add_argument("--meds_reader", required=True,
                   help="meds_reader SubjectDatabase path")
    p.add_argument("--subject_id", type=int, action="append", required=True,
                   help="Subject(s) to explain. Repeat the flag for multiple.")
    p.add_argument("--max_events", type=int, default=400,
                   help="Cap on events included in the trajectory (newest kept).")
    p.add_argument("--narrative_only", action="store_true",
                   help="Print the assembled prompt to stdout and exit without "
                        "calling Claude. Use this to validate inputs first.")
    p.add_argument("--output", default=None,
                   help="Optional path; if set, writes a JSON file with the explanation(s).")
    p.add_argument("--model", default="claude-opus-4-7",
                   help="Anthropic model id")
    args = p.parse_args()

    pretraining_data = pathlib.Path(args.pretraining_data)
    ont_path = pretraining_data / "ontology.pkl"
    print(f"loading ontology from {ont_path}", file=sys.stderr)
    with open(ont_path, "rb") as f:
        ontology = pickle.load(f)

    parquet_path = pathlib.Path(args.reasoning_parquet)

    results = []
    with meds_reader.SubjectDatabase(args.meds_reader, num_threads=1) as db:
        for sid in args.subject_id:
            print(f"\n=== subject {sid} ===", file=sys.stderr)
            meta = load_subject_meta(parquet_path, sid)
            prediction_time = meta["prediction_time"]
            true_label = bool(meta["true_label"])
            predicted_prob = meta.get("predicted_prob")
            outcome_time = meta.get("outcome_time")
            days_to_outcome = meta.get("days_to_outcome")

            subject = db[sid]
            narrative, n_events, n_dates = build_narrative(
                subject, ontology=ontology,
                prediction_time=prediction_time, max_events=args.max_events,
            )

            prompt = build_user_prompt(
                subject_id=sid,
                predicted_prob=predicted_prob,
                prediction_time=prediction_time,
                narrative=narrative,
            )

            print(f"  events kept: {n_events} across {n_dates} dates", file=sys.stderr)
            print(f"  prompt chars: {len(prompt):,}", file=sys.stderr)

            if args.narrative_only:
                print("\n" + "=" * 80)
                print(f"SUBJECT {sid}  --  prompt to be sent to Claude")
                print("=" * 80)
                print(prompt)
                continue

            # Claude API call
            try:
                import anthropic
            except ImportError:
                raise SystemExit(
                    "anthropic SDK not installed. `pip install anthropic` "
                    "or run with --narrative_only to validate prompts first."
                )
            client = anthropic.Anthropic()
            print(f"  calling Anthropic API (model={args.model}) ...", file=sys.stderr)
            with client.messages.stream(
                model=args.model,
                max_tokens=16000,
                thinking={"type": "adaptive"},
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                final = stream.get_final_message()
            text = "".join(b.text for b in final.content if b.type == "text").strip()

            print("\n" + "=" * 80)
            print(f"SUBJECT {sid}  --  Claude explanation")
            print("=" * 80)
            print(text)

            results.append({
                "subject_id": sid,
                "true_label": true_label,
                "predicted_prob": predicted_prob,
                "prediction_time": prediction_time.isoformat() if prediction_time else None,
                "outcome_time": outcome_time.isoformat() if outcome_time else None,
                "days_to_outcome": days_to_outcome,
                "n_events_in_trajectory": n_events,
                "n_dates_in_trajectory": n_dates,
                "explanation": text,
                "usage": {
                    "input_tokens": final.usage.input_tokens,
                    "output_tokens": final.usage.output_tokens,
                    "cache_read_input_tokens": getattr(final.usage, "cache_read_input_tokens", 0),
                    "cache_creation_input_tokens": getattr(final.usage, "cache_creation_input_tokens", 0),
                },
            })

    if args.output and results:
        out = pathlib.Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nwrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()

"""Benchmark Claude on the 5-year TKR-since-KOA prediction task.

For each held-out test subject:
  1. Build a narrative trajectory (demographics + dated event timeline)
     via ``femr.omop_meds_tutorial.meds_narrative``.
  2. Send the narrative to Claude and ask for a calibrated probability
     that the patient will undergo total knee replacement within 5 years
     of the first KOA diagnosis.
  3. Parse Claude's JSON response (probability + rationale) and append it
     to a JSONL file.

Resume-friendly: subjects already present in the output JSONL are skipped.
Concurrency is bounded by ``--max_concurrency`` (per-key rate limits still
apply — the SDK auto-retries 429s).

Modes:
    --predict   (default) Run Claude on subjects and write a JSONL.
    --evaluate            Read the JSONL and compute AUROC / PR-AUC against
                          the ground-truth labels (and MOTOR's predictions).

Typical usage:

    # Start small to validate prompt + cost (~50 subjects, ~$2 on Opus 4.7).
    python -m femr.omop_meds_tutorial.benchmark_claude_tkr predict \\
        --pretraining_data /path/to/motor \\
        --reasoning_parquet  /path/to/reasoning_predictions_long.parquet \\
        --meds_reader        /path/to/meds_reader \\
        --output predictions/claude_opus47_tkr.jsonl \\
        --limit 50 --sample_seed 0

    # Full test set (~3,095 subjects) once the prompt is locked.
    python -m femr.omop_meds_tutorial.benchmark_claude_tkr predict ... --max_concurrency 8

    # Compute metrics on whatever's in the JSONL.
    python -m femr.omop_meds_tutorial.benchmark_claude_tkr evaluate \\
        --reasoning_parquet  /path/to/reasoning_predictions_long.parquet \\
        --predictions predictions/claude_opus47_tkr.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import pickle
import random
import sys
import textwrap
import time
import traceback
from typing import Optional

import meds_reader
import polars as pl
from pydantic import BaseModel, Field

from femr.omop_meds_tutorial.meds_narrative import build_narrative


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a clinical risk-prediction model. Your task is to estimate the
    probability that a patient with newly-diagnosed knee osteoarthritis (KOA)
    will undergo total knee replacement (TKR) within 5 years of their first
    KOA diagnosis.

    You are given:
      - The patient's demographics
      - A chronological timeline of every medical event (diagnoses, procedures,
        medications, visits, labs) recorded prior to and including the first
        KOA diagnosis date

    You must return a single calibrated probability in [0, 1] and a brief
    rationale citing specific evidence from the trajectory.

    # Background on the prediction task
    - The cohort is patients whose first KOA diagnosis (SNOMED OA-of-knee or
      ICD10CM M17.x) is the "prediction time." All trajectory events shown
      to you occur at or before this index event.
    - TKR is defined by CPT4 procedure codes (27445/27447/27486/27487/27488)
      or matching SNOMED codes for total knee arthroplasty.
    - A 60-day washout is applied: TKR events within 60 days of the KOA
      diagnosis are excluded (treated as same-encounter coding rather than
      a true incident outcome).
    - Subjects with a prior partial knee replacement, prior TKR, or a
      "history of TKR" code before first KOA were already excluded from the
      cohort. So every patient you see has no documented prior knee
      arthroplasty in their record.
    - The base rate of 5-year TKR in this cohort is roughly 17%.

    # What good prediction looks like
    Calibrated probabilities, not just rank ordering. Use:
      - Clinical severity signals (meniscal tears, advanced OA imaging,
        chronic NSAID/opioid use, prior intra-articular injections,
        knee-related orthopedic visits)
      - Patient factors (age at diagnosis, BMI proxies like obesity codes,
        diabetes, smoking status — all of which influence both progression
        rate and surgical candidacy)
      - Health-system signals (specialty orthopedic visits, MRI/X-ray
        of knee, physical therapy referrals)
      - Counter-evidence (younger patient, very sparse record, no prior
        knee complaints, recent first knee visit only)

    Do NOT inflate the probability just because a comorbidity is present;
    surgical candidacy depends on the *knee* picture more than overall
    illness burden. Conversely, a patient with established advanced OA,
    chronic pain management, and prior orthopedic visits is a high-risk
    candidate even without comorbidities.
""")


USER_PROMPT_TEMPLATE = textwrap.dedent("""\
    Estimate P(TKR within 5 years of prediction time).

    Prediction time (first KOA diagnosis): {prediction_time}

    {narrative}

    Return JSON with:
      - probability_5y: a single float in [0, 1]
      - rationale: 2-4 sentences citing specific evidence from the trajectory
""")


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

class TKRPrediction(BaseModel):
    probability_5y: float = Field(
        ge=0.0, le=1.0,
        description="Calibrated probability the patient undergoes TKR within 5 years.",
    )
    rationale: str = Field(
        description="2-4 sentence justification citing specific events from the trajectory.",
    )


# ---------------------------------------------------------------------------
# Test-set loading
# ---------------------------------------------------------------------------

def load_test_subjects(parquet_path: pathlib.Path) -> pl.DataFrame:
    """Return one row per test-set subject with the columns we need."""
    df = pl.read_parquet(parquet_path).filter(pl.col("rank") == 1)
    test = (
        df.filter(pl.col("in_test_set"))
          .select([
              "subject_id", "prediction_time", "true_label",
              "predicted_prob", "outcome_time", "days_to_outcome",
          ])
          .sort("subject_id")
    )
    return test


def read_done_subjects(jsonl_path: pathlib.Path) -> set[int]:
    """Read subject_ids already present in the predictions JSONL."""
    if not jsonl_path.exists():
        return set()
    done: set[int] = set()
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = rec.get("subject_id")
            if sid is not None:
                done.add(int(sid))
    return done


# ---------------------------------------------------------------------------
# Per-subject worker
# ---------------------------------------------------------------------------

async def _predict_one(
    *,
    client,
    sema: asyncio.Semaphore,
    model: str,
    row: dict,
    subject,
    ontology,
    max_events: int,
    max_tokens: int,
) -> dict:
    """Call Claude for one subject. Returns a JSONL-ready dict."""
    sid = int(row["subject_id"])
    prediction_time = row["prediction_time"]
    narrative, n_events, n_dates = build_narrative(
        subject, ontology=ontology,
        prediction_time=prediction_time, max_events=max_events,
    )
    user_prompt = USER_PROMPT_TEMPLATE.format(
        prediction_time=prediction_time.date().isoformat(),
        narrative=narrative,
    )

    record = {
        "subject_id": sid,
        "true_label": bool(row["true_label"]),
        "motor_predicted_prob": row.get("predicted_prob"),
        "prediction_time": prediction_time.isoformat() if prediction_time else None,
        "outcome_time": row["outcome_time"].isoformat() if row.get("outcome_time") else None,
        "days_to_outcome": row.get("days_to_outcome"),
        "n_events": n_events,
        "n_dates": n_dates,
        "narrative_chars": len(narrative),
        "model": model,
    }

    t0 = time.time()
    async with sema:
        try:
            response = await client.messages.parse(
                model=model,
                max_tokens=max_tokens,
                thinking={"type": "adaptive"},
                system=[{
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": user_prompt}],
                output_format=TKRPrediction,
            )
            parsed = response.parsed_output
            record.update({
                "claude_predicted_prob": float(parsed.probability_5y) if parsed else None,
                "claude_rationale": parsed.rationale if parsed else None,
                "stop_reason": response.stop_reason,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0),
                "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", 0),
                "latency_s": round(time.time() - t0, 2),
                "error": None,
            })
        except Exception as exc:
            record.update({
                "claude_predicted_prob": None,
                "claude_rationale": None,
                "stop_reason": None,
                "input_tokens": None,
                "output_tokens": None,
                "cache_read_input_tokens": None,
                "cache_creation_input_tokens": None,
                "latency_s": round(time.time() - t0, 2),
                "error": f"{type(exc).__name__}: {exc}",
            })
    return record


# ---------------------------------------------------------------------------
# Predict command
# ---------------------------------------------------------------------------

async def run_predict(args) -> None:
    import anthropic

    pretraining_data = pathlib.Path(args.pretraining_data)
    ont_path = pretraining_data / "ontology.pkl"
    print(f"loading ontology from {ont_path}", file=sys.stderr)
    with open(ont_path, "rb") as f:
        ontology = pickle.load(f)

    parquet_path = pathlib.Path(args.reasoning_parquet)
    print(f"loading test subjects from {parquet_path}", file=sys.stderr)
    test = load_test_subjects(parquet_path)
    print(f"  {test.height:,} test subjects", file=sys.stderr)

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = read_done_subjects(out_path)
    if done:
        print(f"  resuming: {len(done):,} subjects already in {out_path.name}", file=sys.stderr)
        test = test.filter(~pl.col("subject_id").is_in(list(done)))

    if args.subject_id:
        test = test.filter(pl.col("subject_id").is_in(args.subject_id))

    if args.limit and test.height > args.limit:
        if args.sample_seed is not None:
            rng = random.Random(args.sample_seed)
            ids = test["subject_id"].to_list()
            rng.shuffle(ids)
            picked = sorted(ids[: args.limit])
            test = test.filter(pl.col("subject_id").is_in(picked))
        else:
            test = test.head(args.limit)

    if test.height == 0:
        print("nothing to do.", file=sys.stderr)
        return
    print(f"  predicting on {test.height:,} subjects (concurrency={args.max_concurrency})",
          file=sys.stderr)

    client = anthropic.AsyncAnthropic(max_retries=5)
    sema = asyncio.Semaphore(args.max_concurrency)
    rows = test.to_dicts()

    with meds_reader.SubjectDatabase(args.meds_reader, num_threads=1) as db:
        async def _run(r):
            subject = db[int(r["subject_id"])]
            return await _predict_one(
                client=client, sema=sema, model=args.model,
                row=r, subject=subject, ontology=ontology,
                max_events=args.max_events, max_tokens=args.max_tokens,
            )

        tasks = [asyncio.create_task(_run(r)) for r in rows]
        t_start = time.time()
        n_done = 0
        n_ok = 0
        n_err = 0
        with open(out_path, "a") as fout:
            for coro in asyncio.as_completed(tasks):
                rec = await coro
                fout.write(json.dumps(rec, default=str) + "\n")
                fout.flush()
                n_done += 1
                if rec.get("error"):
                    n_err += 1
                else:
                    n_ok += 1
                if n_done % 10 == 0 or n_done == len(tasks):
                    rate = n_done / max(time.time() - t_start, 1e-6)
                    print(
                        f"  [{n_done:>5d}/{len(tasks):>5d}]  ok={n_ok}  err={n_err}  "
                        f"rate={rate:.2f}/s",
                        file=sys.stderr,
                    )

    print(f"\nwrote {out_path}  ({n_ok:,} ok, {n_err:,} errors)", file=sys.stderr)


# ---------------------------------------------------------------------------
# Evaluate command
# ---------------------------------------------------------------------------

def run_evaluate(args) -> None:
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

    preds_path = pathlib.Path(args.predictions)
    if not preds_path.exists():
        raise SystemExit(f"predictions JSONL not found: {preds_path}")
    print(f"loading {preds_path}", file=sys.stderr)

    records: list[dict] = []
    with open(preds_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        raise SystemExit("no records in predictions JSONL")

    df = pl.DataFrame(records, infer_schema_length=None)
    n = df.height
    n_err = df.filter(pl.col("error").is_not_null()).height
    df = df.filter(pl.col("error").is_null() & pl.col("claude_predicted_prob").is_not_null())
    df = df.unique(subset=["subject_id"], keep="last")
    print(f"  loaded {n:,} records ({n_err:,} errors); evaluating on {df.height:,}",
          file=sys.stderr)

    y_true = df["true_label"].to_numpy().astype(int)
    y_claude = df["claude_predicted_prob"].to_numpy().astype(float)
    has_motor = "motor_predicted_prob" in df.columns and df["motor_predicted_prob"].is_not_null().all()
    y_motor = df["motor_predicted_prob"].to_numpy().astype(float) if has_motor else None

    print()
    print(f"  n = {len(y_true):,}")
    print(f"  base rate = {y_true.mean():.4f}  ({int(y_true.sum()):,} positives)")
    print()
    print(f"  {'metric':<18} {'Claude':>10}" + (f" {'MOTOR':>10}" if has_motor else ""))
    print("  " + "-" * (30 if not has_motor else 41))
    metrics = [
        ("ROC-AUC",    roc_auc_score),
        ("PR-AUC",     average_precision_score),
        ("Brier",      brier_score_loss),
    ]
    for name, fn in metrics:
        c = fn(y_true, y_claude)
        if has_motor:
            m = fn(y_true, y_motor)
            print(f"  {name:<18} {c:>10.4f} {m:>10.4f}")
        else:
            print(f"  {name:<18} {c:>10.4f}")

    if args.metrics_out:
        out = pathlib.Path(args.metrics_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n": int(len(y_true)),
            "n_positives": int(y_true.sum()),
            "base_rate": float(y_true.mean()),
            "claude": {
                "roc_auc": float(roc_auc_score(y_true, y_claude)),
                "pr_auc": float(average_precision_score(y_true, y_claude)),
                "brier": float(brier_score_loss(y_true, y_claude)),
            },
        }
        if has_motor:
            payload["motor"] = {
                "roc_auc": float(roc_auc_score(y_true, y_motor)),
                "pr_auc": float(average_precision_score(y_true, y_motor)),
                "brier": float(brier_score_loss(y_true, y_motor)),
            }
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n  metrics written to {out}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- predict ----
    pp = sub.add_parser("predict", help="Run Claude on test-set subjects and write JSONL.")
    pp.add_argument("--pretraining_data", required=True)
    pp.add_argument("--reasoning_parquet", required=True)
    pp.add_argument("--meds_reader", required=True)
    pp.add_argument("--output", required=True, help="JSONL output path (append-mode).")
    pp.add_argument("--model", default="claude-opus-4-7")
    pp.add_argument("--max_events", type=int, default=400)
    pp.add_argument("--max_tokens", type=int, default=2000,
                    help="max_tokens for Claude's response (rationale fits in ~500).")
    pp.add_argument("--max_concurrency", type=int, default=8)
    pp.add_argument("--limit", type=int, default=None,
                    help="Cap test-set size (after resume-skip). Useful for cost control.")
    pp.add_argument("--sample_seed", type=int, default=None,
                    help="If set, pick --limit subjects uniformly at random using this seed.")
    pp.add_argument("--subject_id", type=int, action="append", default=None,
                    help="Restrict to one or more specific subject_ids (repeat the flag).")

    # ---- evaluate ----
    pe = sub.add_parser("evaluate", help="Compute AUROC/PR-AUC/Brier from the JSONL.")
    pe.add_argument("--predictions", required=True)
    pe.add_argument("--reasoning_parquet", required=False,
                    help="(Unused now — labels are already in the JSONL; kept for parity.)")
    pe.add_argument("--metrics_out", default=None, help="Optional JSON path for the metrics summary.")

    args = p.parse_args()
    if args.cmd == "predict":
        asyncio.run(run_predict(args))
    elif args.cmd == "evaluate":
        run_evaluate(args)


if __name__ == "__main__":
    main()

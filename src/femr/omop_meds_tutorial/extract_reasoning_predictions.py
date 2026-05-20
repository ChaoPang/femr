"""Extract a tidy long-format parquet of MOTOR+reasoning attributions per subject.

Joins together:
  * motor/features/<label>_motor.pkl   — reasoning_token_ids + reasoning_weights + subject_ids
  * motor/labels/<label>.parquet       — boolean_value (true label)
  * motor/results/<label>/motor/test_predictions/*.parquet
                                       — predicted_boolean_probability (test subjects only)
  * motor/tokenizer/dictionary.msgpack — vocab → code_string mapping
  * motor/ontology.pkl                 — code_string → Athena concept_name
  * (optional) meds_reader SubjectDatabase
                                       — first qualifying TKR datetime per subject

Subjects not in the held-out test split get predicted_prob = NULL. Subjects who
never had a qualifying TKR (most label = False subjects) get outcome_time = NULL.

Output schema (long format, one row per (subject, rank)):
  subject_id           int64
  prediction_time      datetime[us]
  true_label           bool
  predicted_prob       float32 (nullable)
  in_test_set          bool
  outcome_time         datetime[us] (nullable) — first TKR after prediction_time+washout
  days_to_outcome      int32 (nullable)        — (outcome_time - prediction_time).days
  rank                 int32     # 1..k, sorted by weight desc within a subject
  token_id             int32
  weight               float32
  code_string          string    # e.g. "ATC/V08A", "<numeric ...>"
  description          string    # e.g. "X-RAY CONTRAST MEDIA, IODINATED"
"""
from __future__ import annotations

import argparse
import datetime as _datetime
import functools
import math
import pathlib
import pickle
import time
from typing import Iterable, Iterator, Optional

import msgpack
import numpy as np
import pandas as pd
import polars as pl


def _outcome_time_worker(
    subjects, *, tkr_codes: set, koa_codes: set, washout: _datetime.timedelta,
):
    """For each subject return (subject_id, first_qualifying_tkr_time).

    Mirrors TKRSinceKOALabeler's outcome detection: first KOA event marks the
    prediction time, and the outcome is the first TKR code event strictly
    after first_koa_time + washout. Returns ``None`` outcome for subjects
    without a qualifying TKR.
    """
    out = []
    for s in subjects:
        first_koa = None
        outcome = None
        for e in s.events:
            if e.time is None:
                continue
            if first_koa is None:
                if e.code in koa_codes:
                    first_koa = e.time
                continue
            if e.code in tkr_codes and e.time > first_koa + washout:
                outcome = e.time
                break
        out.append((int(s.subject_id), outcome))
    return out


def _vocab_label(entry: dict) -> str:
    if entry["type"] == "code":
        return entry["code_string"]
    if entry["type"] == "numeric":
        start = entry.get("val_start")
        end = entry.get("val_end")
        def fmt(x):
            if x is None: return "?"
            if math.isinf(x): return "-inf" if x < 0 else "inf"
            return f"{x:g}"
        return f"<numeric [{fmt(start)},{fmt(end)})>"
    return f"<{entry.get('type', '?')}>"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pretraining_data", required=True,
                   help="Folder with motor artifacts (features/, labels/, results/, tokenizer/, ontology.pkl)")
    p.add_argument("--cohort_label", default="tkr_since_koa",
                   help="Label name (matches features/<label>_motor.pkl etc.)")
    p.add_argument("--output", required=True,
                   help="Output parquet path")
    p.add_argument("--results_subdir", default="motor",
                   help="Subdirectory under results/<label>/ holding test_predictions/ "
                        "(use 'motor_original' to point at the non-reasoning baseline)")
    p.add_argument("--meds_reader", default=None,
                   help="Optional: meds_reader SubjectDatabase path. When set, "
                        "scans the database to populate outcome_time + days_to_outcome "
                        "(first qualifying TKR after first_koa + tkr_washout). "
                        "When unset, those columns are null.")
    p.add_argument("--tkr_washout_days", type=int, default=60,
                   help="Days after first KOA before a TKR counts as a qualifying outcome "
                        "(must match the value used by TKRSinceKOALabeler).")
    p.add_argument("--num_threads", type=int, default=16,
                   help="Worker threads for the meds_reader scan.")
    args = p.parse_args()

    pretraining_data = pathlib.Path(args.pretraining_data)
    label = args.cohort_label

    features_path = pretraining_data / "features" / f"{label}_motor.pkl"
    labels_path = pretraining_data / "labels" / f"{label}.parquet"
    preds_dir = pretraining_data / "results" / label / args.results_subdir / "test_predictions"
    tok_path = pretraining_data / "tokenizer" / "dictionary.msgpack"
    ont_path = pretraining_data / "ontology.pkl"
    out = pathlib.Path(args.output)

    # Load
    print(f"loading {features_path}")
    with open(features_path, "rb") as f:
        feats = pickle.load(f)
    if "reasoning_token_ids" not in feats or "reasoning_weights" not in feats:
        raise SystemExit(
            f"{features_path} has no reasoning_token_ids/reasoning_weights — "
            "rerun generate_motor_features with a model that has the reasoning layer."
        )
    sids: np.ndarray = feats["subject_ids"]
    fts: np.ndarray = feats["feature_times"]
    rt_ids: np.ndarray = feats["reasoning_token_ids"]
    rt_w: np.ndarray = feats["reasoning_weights"]
    n_subj, k = rt_ids.shape
    print(f"  {n_subj:,} subjects × top-{k} reasoning tokens")

    print(f"loading {labels_path}")
    labels_df = pd.read_parquet(labels_path)
    label_lookup = dict(zip(labels_df.subject_id.values,
                            labels_df.boolean_value.astype(bool).values))

    pred_lookup: dict[int, float] = {}
    if preds_dir.exists():
        pred_files = list(preds_dir.glob("*.parquet"))
        print(f"loading {len(pred_files)} prediction files from {preds_dir}")
        for pf in pred_files:
            d = pl.read_parquet(pf)
            for sid, prob in zip(d["subject_id"].to_list(),
                                 d["predicted_boolean_probability"].to_list()):
                pred_lookup[int(sid)] = float(prob)
        print(f"  {len(pred_lookup):,} test-set predictions loaded")
    else:
        print(f"  no test_predictions dir at {preds_dir}; predicted_prob will be null")

    print(f"loading vocab from {tok_path}")
    vocab = msgpack.unpackb(tok_path.read_bytes(), raw=False)["vocab"]
    idx2label = {i: _vocab_label(e) for i, e in enumerate(vocab)}

    print(f"loading ontology from {ont_path}")
    with open(ont_path, "rb") as f:
        ontology = pickle.load(f)
    def describe(code: str) -> str:
        if code.startswith("<"): return ""
        return ontology.get_description(code) or ""

    # Optional: outcome_time per subject from meds_reader
    outcome_lookup: dict[int, _datetime.datetime] = {}
    if args.meds_reader:
        import meds_reader
        from femr.labelers.koa_tkr import KOA_CODES, TKR_CODES
        washout = _datetime.timedelta(days=args.tkr_washout_days)
        # Only scan subjects that actually have labels (saves time on large DBs)
        labeled_ids = set(int(s) for s in sids.tolist())
        print(f"scanning {args.meds_reader} for outcome_time on {len(labeled_ids):,} labeled subjects ...")
        t0 = time.time()
        worker = functools.partial(
            _outcome_time_worker,
            tkr_codes=set(TKR_CODES), koa_codes=set(KOA_CODES), washout=washout,
        )
        with meds_reader.SubjectDatabase(args.meds_reader, num_threads=args.num_threads) as db:
            for chunk in db.map(worker):
                for sid, outcome in chunk:
                    if sid in labeled_ids and outcome is not None:
                        outcome_lookup[sid] = outcome
        print(f"  scanned in {time.time()-t0:.1f}s; "
              f"{len(outcome_lookup):,} subjects have an outcome_time")
    else:
        print("--meds_reader not provided; outcome_time will be null")

    # Build long-format dataframe
    print("building long-format table ...")
    t0 = time.time()
    order = np.argsort(-rt_w, axis=1)  # sort each row by weight desc
    sorted_ids = np.take_along_axis(rt_ids, order, axis=1).astype(np.int32)
    sorted_w = np.take_along_axis(rt_w, order, axis=1).astype(np.float32)

    n_rows = n_subj * k
    flat_subj = np.repeat(sids, k).astype(np.int64)
    flat_ft = np.repeat(fts, k)  # datetime64
    flat_label = np.array(
        [label_lookup.get(int(s), False) for s in sids], dtype=bool
    ).repeat(k)
    in_test = np.array([int(s) in pred_lookup for s in sids], dtype=bool)
    flat_in_test = in_test.repeat(k)
    flat_pred = np.array(
        [pred_lookup.get(int(s), np.nan) for s in sids], dtype=np.float32
    ).repeat(k)
    flat_rank = np.tile(np.arange(1, k + 1, dtype=np.int32), n_subj)
    flat_tid = sorted_ids.ravel()
    flat_w = sorted_w.ravel()
    flat_code = np.array([idx2label.get(int(t), "?") for t in flat_tid], dtype=object)
    flat_desc = np.array([describe(c) for c in flat_code], dtype=object)
    # Outcome time + days_to_outcome per subject, repeated k times.
    NaT64 = np.datetime64("NaT", "us")
    per_subj_outcome = np.array(
        [
            np.datetime64(outcome_lookup[int(s)], "us") if int(s) in outcome_lookup else NaT64
            for s in sids
        ],
        dtype="datetime64[us]",
    )
    flat_outcome = np.repeat(per_subj_outcome, k)
    # days_to_outcome = (outcome - prediction_time).days where both are present
    per_subj_days = np.full(n_subj, np.iinfo(np.int32).min, dtype=np.int64)
    pred_dt = np.asarray(fts, dtype="datetime64[us]")
    for i, s in enumerate(sids):
        if int(s) in outcome_lookup:
            delta = np.datetime64(outcome_lookup[int(s)], "us") - pred_dt[i]
            per_subj_days[i] = int(delta / np.timedelta64(1, "D"))
    flat_days = np.repeat(per_subj_days, k).astype(np.int64)
    days_missing_sentinel = np.iinfo(np.int32).min

    print(f"  built {n_rows:,} rows in {time.time()-t0:.1f}s")

    df = pl.DataFrame({
        "subject_id":      flat_subj,
        "prediction_time": pd.to_datetime(flat_ft).astype("datetime64[us]"),
        "true_label":      flat_label,
        "predicted_prob":  flat_w * 0 + flat_pred,  # keep dtype float32, NaN for missing
        "in_test_set":     flat_in_test,
        "outcome_time":    flat_outcome,
        "days_to_outcome": flat_days,
        "rank":            flat_rank,
        "token_id":        flat_tid,
        "weight":          flat_w,
        "code_string":     flat_code.tolist(),
        "description":     flat_desc.tolist(),
    })
    # NaT -> null on the datetime, sentinel -> null on the days column
    df = df.with_columns(
        pl.when(pl.col("days_to_outcome") == days_missing_sentinel)
          .then(None).otherwise(pl.col("days_to_outcome").cast(pl.Int32))
          .alias("days_to_outcome"),
    )
    # NaN -> null for the prediction column
    df = df.with_columns(
        pl.when(pl.col("predicted_prob").is_nan()).then(None).otherwise(pl.col("predicted_prob")).alias("predicted_prob")
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    sz_mb = out.stat().st_size / 1e6
    print(f"wrote {out}  rows={df.height:,}  size={sz_mb:.1f} MB")
    print()
    print("schema:")
    print(df.schema)
    print()
    print("first 5 rows:")
    print(df.head(5))


if __name__ == "__main__":
    main()

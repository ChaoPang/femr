"""Extract a tidy long-format parquet of (subject, prediction_time, true_label,
predicted_prob, rank, token_id, weight, code_string, description) for every
labeled subject in the MOTOR+reasoning features pickle.

Joins together:
  * motor/features/<label>_motor.pkl   — reasoning_token_ids + reasoning_weights + subject_ids
  * motor/labels/<label>.parquet       — boolean_value (true label)
  * motor/results/<label>/motor/test_predictions/*.parquet
                                       — predicted_boolean_probability (test subjects only)
  * motor/tokenizer/dictionary.msgpack — vocab → code_string mapping
  * motor/ontology.pkl                 — code_string → Athena concept_name

Subjects not in the held-out test split get predicted_prob = NULL.

Output schema (long format, one row per (subject, rank)):
  subject_id           int64
  prediction_time      datetime[us]
  true_label           bool
  predicted_prob       float32 (nullable)
  in_test_set          bool
  rank                 int32     # 1..32, sorted by weight desc within a subject
  token_id             int32
  weight               float32
  code_string          string    # e.g. "ATC/V08A", "<numeric ...>"
  description          string    # e.g. "X-RAY CONTRAST MEDIA, IODINATED"
"""
from __future__ import annotations

import argparse
import math
import pathlib
import pickle
import time
from typing import Optional

import msgpack
import numpy as np
import pandas as pd
import polars as pl


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
    print(f"  built {n_rows:,} rows in {time.time()-t0:.1f}s")

    df = pl.DataFrame({
        "subject_id":      flat_subj,
        "prediction_time": pd.to_datetime(flat_ft).astype("datetime64[us]"),
        "true_label":      flat_label,
        "predicted_prob":  flat_w * 0 + flat_pred,  # keep dtype float32, NaN for missing
        "in_test_set":     flat_in_test,
        "rank":            flat_rank,
        "token_id":        flat_tid,
        "weight":          flat_w,
        "code_string":     flat_code.tolist(),
        "description":     flat_desc.tolist(),
    })
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

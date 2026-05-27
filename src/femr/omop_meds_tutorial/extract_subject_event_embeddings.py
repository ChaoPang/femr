"""Extract per-event MOTOR embeddings for an entire split of subjects.

Iterates subjects in shards (default 1,000/shard), runs MOTOR forward pass
once per unique (subject_id, timestamp), and writes parquet shards keyed by
shard index. Idempotent: skips shards whose output parquet already exists,
so a crashed run can be resumed by re-running the same command.

Output schema (one row per source event):
  subject_id        int64        -- meds_reader subject_id
  PersonId          string|null  -- original Truveta UUID (left-joined from
                                   meds_reader/metadata/person_id_mapping.parquet)
  source_record_id  string       -- per-event UUID from the source EHR row
  encounter_id      string|null  -- visit-level UUID (null for demographics)
  timestamp         timestamp[us]
  code              string       -- e.g. "SNOMED/123", "ENCOUNTER//Ambulatory"
  position          int32        -- per-subject 0..N-1 index in time-sorted
                                   order; events at the same timestamp share
                                   the same position AND the same embedding
  embedding         fixed_size_list<float16>[hidden_size]

The model produces one hidden state per unique (subject_id, timestamp); when
two source events share a timestamp, both rows in the output carry the same
embedding (and the same position).

Usage:
  python -m femr.omop_meds_tutorial.extract_subject_event_embeddings \\
      --pretraining_data /home/sagemaker-user/truveta_koa_full \\
      --meds_reader /home/sagemaker-user/truveta_koa_full/meds_reader \\
      --model_dir /home/sagemaker-user/truveta_koa_full/tmp_trainer_plain_lr2e-4/checkpoint-163368 \\
      --output_dir /home/sagemaker-user/truveta_koa_full/event_embeddings \\
      --split test \\
      --subjects_per_shard 1000 \\
      --tokens_per_batch 16384 \\
      --num_proc 10 \\
      --dtype float16

Add ``--max_shards N`` (or ``--end_shard N``) to limit total work, e.g. for
processing the first 10% of the chosen split: 44 shards * 1,000 subjects
each = first 44,000 test subjects.
"""
from __future__ import annotations

import argparse
import pathlib
import pickle
import time
from typing import List, Tuple

import meds
import meds_reader
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch

import femr.models.transformer

from .generate_labels import create_omop_meds_tutorial_arg_parser
from .generate_motor_features import resolve_model_dir


def create_arg_parser() -> argparse.ArgumentParser:
    args = create_omop_meds_tutorial_arg_parser()
    args.add_argument(
        "--model_dir",
        dest="model_dir",
        default="motor_model",
        help="Directory holding the pretrained MOTOR weights + config + "
             "dictionary.msgpack. Absolute path used as-is; relative path is "
             "joined against --pretraining_data. Default: motor_model.",
    )
    args.add_argument(
        "--output_dir",
        dest="output_dir",
        required=True,
        help="Directory to write parquet shards into. One file per shard; "
             "naming pattern: embeddings_{shard_idx:06d}.parquet.",
    )
    args.add_argument(
        "--split",
        dest="split",
        default="test",
        help="Which subjects to process (matches values in main_split.csv: "
             "typically 'train' or 'test'). Default: test.",
    )
    args.add_argument(
        "--subjects_per_shard",
        dest="subjects_per_shard",
        type=int,
        default=1000,
        help="Number of subjects per output parquet shard. Default: 1,000.",
    )
    args.add_argument(
        "--tokens_per_batch",
        dest="tokens_per_batch",
        type=int,
        default=16384,
        help="MOTOR packed-batch token budget. Default: 16,384.",
    )
    args.add_argument(
        "--num_proc",
        dest="num_proc",
        type=int,
        default=10,
        help="Number of meds_reader workers. Default: 10.",
    )
    args.add_argument(
        "--dtype",
        dest="dtype",
        default="float16",
        choices=["float16", "float32"],
        help="Embedding storage dtype. Default: float16 (~1.5 KB/row).",
    )
    args.add_argument(
        "--start_shard",
        dest="start_shard",
        type=int,
        default=0,
        help="First shard index to process (inclusive). Default: 0.",
    )
    args.add_argument(
        "--end_shard",
        dest="end_shard",
        type=int,
        default=None,
        help="One-past-the-last shard index to process. Default: all shards.",
    )
    args.add_argument(
        "--max_shards",
        dest="max_shards",
        type=int,
        default=None,
        help="Convenience: cap how many shards this invocation will process "
             "(applied after --start_shard).",
    )
    args.add_argument(
        "--compression",
        dest="compression",
        default="zstd",
        help="Parquet compression codec. Default: zstd.",
    )
    return args


def _collect_shard_inputs(
    db: meds_reader.SubjectDatabase, shard_sids: List[int]
) -> Tuple[List[meds.Label], pl.DataFrame]:
    """Build the label list (one per unique (subject_id, timestamp))
    and the per-event metadata table for a shard.

    The events table has one row per source event; the position column
    is the 0..N-1 index of that event's timestamp within the subject's
    time-sorted sequence (events sharing a timestamp share a position).
    """
    labels: List[meds.Label] = []
    events_rows: List[dict] = []
    for sid in shard_sids:
        s = db[int(sid)]
        # Time-sorted assignment of position index, sharing across events
        # at the same timestamp. meds_reader events are already sorted by
        # time but we don't rely on that.
        time_to_pos: dict = {}
        for e in s.events:
            if e.time is None:
                continue
            if e.time not in time_to_pos:
                time_to_pos[e.time] = len(time_to_pos)
                labels.append(
                    meds.Label(
                        subject_id=int(sid),
                        prediction_time=e.time,
                        boolean_value=False,  # dummy: ignored by compute_features
                    )
                )
            events_rows.append(
                {
                    "subject_id": int(sid),
                    "source_record_id": e.source_record_id,
                    "encounter_id": e.encounter_id,
                    "timestamp": pd.Timestamp(e.time),
                    "code": e.code,
                    "position": time_to_pos[e.time],
                }
            )
    events_df = pl.DataFrame(events_rows)
    return labels, events_df


def _write_shard(
    shard_path: pathlib.Path,
    events_df: pl.DataFrame,
    features: dict,
    sid_to_pid: dict,
    hidden_size: int,
    dtype: str,
    compression: str,
) -> None:
    """Join per-event metadata with embeddings and write the shard parquet."""
    # Build the (subject_id, timestamp_us) -> embedding lookup
    sids = np.asarray(features["subject_ids"]).astype(np.int64)
    times = np.asarray(features["feature_times"], dtype="datetime64[us]").view(np.int64)
    embs = np.asarray(features["features"], dtype=np.float32)
    assert embs.shape[0] == sids.shape[0] == times.shape[0]
    assert embs.shape[1] == hidden_size

    # Pack the lookup into a polars frame so the join is vectorized
    lookup = pl.DataFrame(
        {
            "subject_id": sids,
            "_ts_us": times,
            "_emb_idx": np.arange(embs.shape[0], dtype=np.int64),
        }
    )

    df = events_df.with_columns(
        pl.col("timestamp").cast(pl.Datetime("us")).alias("timestamp"),
    )
    df = df.with_columns(
        pl.col("timestamp").cast(pl.Int64).alias("_ts_us"),
    )
    df = df.join(lookup, on=["subject_id", "_ts_us"], how="left").drop("_ts_us")
    if df.select(pl.col("_emb_idx").is_null().any()).item():
        n_missing = int(df.select(pl.col("_emb_idx").is_null().sum()).item())
        raise RuntimeError(
            f"{n_missing:,} event rows did not match a (subject_id, timestamp) embedding; "
            "this should not happen if every event time was emitted as a label."
        )

    # Gather embeddings in event-row order
    emb_idx = df["_emb_idx"].to_numpy().astype(np.int64)
    emb_target_dtype = np.float16 if dtype == "float16" else np.float32
    gathered = embs[emb_idx].astype(emb_target_dtype, copy=False)

    # Build pyarrow table with a fixed-size list<halffloat | float> column
    base = df.drop("_emb_idx")
    # Join PersonId (left): null when subject not in mapping (unlikely)
    base = base.with_columns(
        pl.col("subject_id").map_elements(
            lambda s: sid_to_pid.get(int(s)), return_dtype=pl.Utf8
        ).alias("PersonId")
    )
    # Reorder columns to match docstring schema
    base = base.select(
        ["subject_id", "PersonId", "source_record_id", "encounter_id",
         "timestamp", "code", "position"]
    )
    arrow_table = base.to_arrow()
    elem_type = pa.float16() if dtype == "float16" else pa.float32()
    emb_column = pa.FixedSizeListArray.from_arrays(
        pa.array(gathered.reshape(-1).tolist(), type=elem_type),
        hidden_size,
    )
    arrow_table = arrow_table.append_column("embedding", emb_column)

    shard_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = shard_path.with_suffix(".parquet.tmp")
    pq.write_table(arrow_table, tmp_path, compression=compression)
    tmp_path.replace(shard_path)


def main() -> None:
    args = create_arg_parser().parse_args()
    pretraining_data = pathlib.Path(args.pretraining_data)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Split filter — must match values in main_split.csv (typically train/test).
    split_path = pretraining_data / "main_split.csv"
    split_df = pd.read_csv(split_path)
    split_subjects = sorted(
        int(s) for s in split_df.loc[split_df["split_name"] == args.split, "subject_id"]
    )
    if not split_subjects:
        raise SystemExit(
            f"No subjects in '{args.split}' split at {split_path}; "
            f"available values: {sorted(split_df['split_name'].unique().tolist())}"
        )
    print(f"split='{args.split}': {len(split_subjects):,} subjects")

    # PersonId mapping: meds_reader subject_id -> Truveta UUID
    mapping_path = (
        pretraining_data / "meds_reader" / "metadata" / "person_id_mapping.parquet"
    )
    if mapping_path.exists():
        mapping = pl.read_parquet(mapping_path)
        sid_to_pid = dict(zip(
            mapping["subject_id"].to_list(),
            mapping["PersonId"].to_list(),
        ))
        print(f"loaded PersonId mapping ({len(sid_to_pid):,} subjects)")
    else:
        print(f"WARN: {mapping_path} not found; PersonId column will be null")
        sid_to_pid = {}

    # Ontology — required by compute_features for hierarchical tokenization
    ontology_path = pretraining_data / "ontology.pkl"
    print(f"loading ontology from {ontology_path}")
    with open(ontology_path, "rb") as f:
        ontology = pickle.load(f)

    # Resolve the MOTOR model dir; sanity-check shape from config
    model_dir_path = resolve_model_dir(pretraining_data, args.model_dir)
    if not (model_dir_path / "model.safetensors").exists():
        raise SystemExit(f"No model.safetensors at {model_dir_path}")
    # Read hidden size from config so we know the embedding dimension
    import json
    cfg = json.loads((model_dir_path / "config.json").read_text())
    hidden_size = int(
        cfg.get("transformer_config", cfg).get("hidden_size", 768)
    )
    print(f"model_dir={model_dir_path}  hidden_size={hidden_size}")

    # Shard plan
    n_total_shards = (
        len(split_subjects) + args.subjects_per_shard - 1
    ) // args.subjects_per_shard
    end_shard = args.end_shard if args.end_shard is not None else n_total_shards
    if args.max_shards is not None:
        end_shard = min(end_shard, args.start_shard + args.max_shards)
    end_shard = min(end_shard, n_total_shards)
    print(
        f"shards: {n_total_shards:,} total at {args.subjects_per_shard}/shard; "
        f"this run: [{args.start_shard}, {end_shard})"
    )

    with meds_reader.SubjectDatabase(args.meds_reader, num_threads=args.num_proc) as db:
        for shard_idx in range(args.start_shard, end_shard):
            shard_path = output_dir / f"embeddings_{shard_idx:06d}.parquet"
            if shard_path.exists():
                print(f"[shard {shard_idx:>4}/{n_total_shards}] exists, skip: {shard_path.name}")
                continue

            t0 = time.time()
            shard_start = shard_idx * args.subjects_per_shard
            shard_end = min(shard_start + args.subjects_per_shard, len(split_subjects))
            shard_sids = split_subjects[shard_start:shard_end]
            print(
                f"[shard {shard_idx:>4}/{n_total_shards}] "
                f"subjects [{shard_start:,}, {shard_end:,})  n={len(shard_sids):,}"
            )

            labels, events_df = _collect_shard_inputs(db, shard_sids)
            n_events = events_df.height
            print(
                f"  collected {len(labels):,} unique (subject_id, timestamp) labels "
                f"and {n_events:,} events  ({time.time()-t0:.1f}s)"
            )

            t1 = time.time()
            features = femr.models.transformer.compute_features(
                db=db,
                model_path=str(model_dir_path),
                labels=labels,
                ontology=ontology,
                device=torch.device("cuda"),
                tokens_per_batch=args.tokens_per_batch,
                num_proc=args.num_proc,
                observation_window=None,
            )
            print(
                f"  forward pass: {features['features'].shape[0]:,} embeddings "
                f"in {(time.time()-t1)/60:.1f} min"
            )

            t2 = time.time()
            _write_shard(
                shard_path,
                events_df,
                features,
                sid_to_pid,
                hidden_size=hidden_size,
                dtype=args.dtype,
                compression=args.compression,
            )
            sz_mb = shard_path.stat().st_size / 1e6
            print(
                f"  wrote {shard_path}  rows={n_events:,}  size={sz_mb:.1f} MB  "
                f"({time.time()-t2:.1f}s)  shard_total={(time.time()-t0)/60:.1f} min"
            )


if __name__ == "__main__":
    main()

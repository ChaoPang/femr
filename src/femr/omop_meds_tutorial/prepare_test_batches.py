"""Generate test_batches for a held-out split of subjects.

Mirrors the val_batches step in prepare_motor.py, but operates on the
held-out cohort identified by --pretraining_data/main_split.csv. Uses the
existing tokenizer, motor_task and ontology that prepare_motor.py wrote
during pretraining preparation; no retraining or re-fitting happens.

Output: <pretraining_data>/test_batches/ (an Arrow dataset, same layout as
train_batches/ and val_batches/).
"""
import argparse
import pathlib
import pickle
import time

import meds_reader
import pandas as pd
import femr.models.tokenizer
import femr.models.processor


def main(args):
    pretraining_data = pathlib.Path(args.pretraining_data)
    output_dir = pathlib.Path(args.output_dir) if args.output_dir else pretraining_data
    output_dir.mkdir(parents=True, exist_ok=True)
    batches_path = output_dir / "test_batches"
    if batches_path.exists():
        raise SystemExit(f"refusing to overwrite existing {batches_path}")

    print("loading ontology...", flush=True)
    with open(pretraining_data / "ontology.pkl", "rb") as f:
        ontology = pickle.load(f)

    print("loading tokenizer...", flush=True)
    tokenizer = femr.models.tokenizer.HierarchicalTokenizer.from_pretrained(
        pretraining_data / "tokenizer", ontology=ontology,
    )

    print("loading motor_task...", flush=True)
    with open(pretraining_data / "motor_task.pkl", "rb") as f:
        motor_task = pickle.load(f)

    print("loading main_split.csv...", flush=True)
    split_df = pd.read_csv(pretraining_data / "main_split.csv")
    test_ids = sorted(
        int(s) for s in split_df.loc[split_df["split_name"] == args.split_name, "subject_id"]
    )
    if not test_ids:
        raise SystemExit(
            f"no subjects in split '{args.split_name}'; "
            f"available: {sorted(split_df['split_name'].unique().tolist())}"
        )
    print(f"selected {len(test_ids):,} subjects from split '{args.split_name}'", flush=True)

    processor = femr.models.processor.FEMRBatchProcessor(tokenizer, motor_task)

    print(f"opening meds_reader (num_threads={args.num_threads})...", flush=True)
    with meds_reader.SubjectDatabase(args.meds_reader, num_threads=args.num_threads) as db:
        test_db = db.filter(test_ids)
        print(f"converting dataset (tokens_per_batch={args.tokens_per_batch})...", flush=True)
        t0 = time.time()
        test_batches = processor.convert_dataset(
            test_db,
            tokens_per_batch=args.tokens_per_batch,
            min_subjects_per_batch=1,
            num_proc=args.num_threads,
        )
        print(f"convert_dataset done in {time.time() - t0:.0f}s; saving to {batches_path}",
              flush=True)
        test_batches.set_format("pt")
        test_batches.save_to_disk(str(batches_path))

    print(f"DONE: wrote {batches_path}", flush=True)


def create_omop_meds_tutorial_argparser():
    parser = argparse.ArgumentParser(description="Arguments for preparing held-out test batches")
    parser.add_argument(
        "--pretraining_data",
        dest="pretraining_data",
        action="store",
        required=True,
        help="Directory with tokenizer/, motor_task.pkl, ontology.pkl, and main_split.csv "
             "(written by prepare_motor.py).",
    )
    parser.add_argument(
        "--meds_reader",
        dest="meds_reader",
        action="store",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        action="store",
        required=False,
        default=None,
        help="Destination dir for test_batches/. Defaults to --pretraining_data so the "
             "layout mirrors train_batches/ and val_batches/.",
    )
    parser.add_argument(
        "--split_name",
        dest="split_name",
        action="store",
        required=False,
        default="test",
        help="Value in main_split.csv to select; default 'test' = MEDS held_out.",
    )
    parser.add_argument(
        "--num_threads",
        dest="num_threads",
        action="store",
        required=False,
        type=int,
        default=16,
    )
    parser.add_argument(
        "--tokens_per_batch",
        dest="tokens_per_batch",
        action="store",
        required=False,
        type=int,
        default=16384,
    )
    return parser


if __name__ == "__main__":
    main(create_omop_meds_tutorial_argparser().parse_args())

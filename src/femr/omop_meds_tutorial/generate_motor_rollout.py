"""Generate attention rollouts for a labeled cohort using a trained MOTOR model.

Runs `compute_features(..., return_rollout=True)` on the test-split subset of
the given cohort and saves a rollout pickle alongside the features directory.

This is intentionally a separate process from `generate_motor_features` so that
(a) feature extraction can run without the heavier rollout pass, and (b) the
rollout can be re-computed (e.g., with a different top_k) without touching the
features pkl.
"""
import os
import glob
import datetime
from pathlib import Path
import pickle
import pandas as pd
import torch

import meds
import meds_reader
import femr.models.transformer
import femr.splits

from .generate_labels import create_omop_meds_tutorial_arg_parser, LABEL_NAMES


def create_arg_parser():
    args = create_omop_meds_tutorial_arg_parser()
    args.add_argument(
        "--num_proc",
        dest="num_proc",
        type=int,
        default=6,
        help="Number of processes to use",
    )
    args.add_argument(
        "--tokens_per_batch",
        dest="tokens_per_batch",
        type=int,
        default=4096,
        help="Tokens per batch (keep small; rollout materializes full attention matrices)",
    )
    args.add_argument(
        "--cohort_dir",
        dest="cohort_dir",
        default=None,
    )
    args.add_argument(
        "--observation_window",
        dest="observation_window",
        type=int,
        default=None,
        help="Observation window for feature extraction",
    )
    args.add_argument(
        "--rollout_top_k",
        dest="rollout_top_k",
        type=int,
        default=50,
        help="Number of top input positions to return per label in the attention rollout",
    )
    args.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Recompute rollout even if the output pkl exists",
    )
    args.add_argument(
        "--model_dir",
        dest="model_dir",
        default="motor_model",
        help="Directory holding the pretrained MOTOR weights + config + dictionary.msgpack. "
             "If a relative path, it is resolved against --pretraining_data. Defaults to "
             "the legacy `motor_model` subdir for backwards compatibility.",
    )
    args.add_argument(
        "--variant",
        dest="variant",
        default=None,
        help="Optional suffix appended to the rollout pkl basename: produces "
             "features/<label>_motor_<variant>_rollout.pkl. When unset, the legacy "
             "`<label>_motor_rollout` path is used.",
    )
    return args


def read_recursive_parquet(root_dir):
    files = glob.glob(os.path.join(root_dir, "**", "*.parquet"), recursive=True)
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def get_rollout_output_name(
    label_name: str,
    observation_window=None,
    variant=None,
) -> str:
    """Build the canonical basename for a motor rollout pkl.

    Pattern (suffix order keeps backwards compat when ``variant`` is None):
        <label>_motor_rollout                                  # default
        <label>_motor_<observation_window>_rollout             # legacy windowed
        <label>_motor_<variant>_rollout                        # variant only (new)
        <label>_motor_<observation_window>_<variant>_rollout   # both (new)
    """
    base = label_name + "_motor"
    if observation_window:
        base = base + f"_{observation_window}"
    if variant:
        base = base + f"_{variant}"
    return base + "_rollout"


def main():
    args = create_arg_parser().parse_args()
    pretraining_data = Path(args.pretraining_data)
    features_path = pretraining_data / "features"
    features_path.mkdir(exist_ok=True, parents=True)

    with open(pretraining_data / "ontology.pkl", "rb") as f:
        ontology = pickle.load(f)

    # Resolve cohort
    labels_to_process = LABEL_NAMES
    if args.cohort_dir is not None:
        if os.path.isdir(args.cohort_dir):
            label_name = os.path.basename(os.path.normpath(args.cohort_dir))
            cohort = read_recursive_parquet(args.cohort_dir)
        else:
            label_name = os.path.basename(os.path.splitext(args.cohort_dir)[0])
            ext = os.path.splitext(args.cohort_dir)[1].lower()
            if ext == ".parquet":
                cohort = pd.read_parquet(args.cohort_dir)
            elif ext == ".csv":
                cohort = pd.read_csv(args.cohort_dir)
            else:
                raise RuntimeError(f"Unknown cohort file extension: {ext}")
        if len(cohort) > 0 and isinstance(cohort.prediction_time.iloc[0], datetime.date):
            cohort["prediction_time"] = pd.to_datetime(cohort["prediction_time"])
        (pretraining_data / "labels").mkdir(exist_ok=True, parents=True)
        cohort.to_parquet(pretraining_data / "labels" / (label_name + ".parquet"))
        labels_to_process = [label_name]

    split_path = pretraining_data / "main_split.csv"
    if not split_path.exists():
        raise RuntimeError(f"main_split.csv not found at {split_path}")
    main_split = femr.splits.SubjectSplit.load_from_csv(str(split_path))
    test_subject_ids = set(main_split.test_subject_ids)

    # Resolve model dir (lazy import to avoid circular dep at module import time).
    from .generate_motor_features import resolve_model_dir
    model_dir_path = resolve_model_dir(pretraining_data, args.model_dir)

    with meds_reader.SubjectDatabase(args.meds_reader, num_threads=6) as database:
        for label_name in labels_to_process:
            rollout_name = get_rollout_output_name(
                label_name, args.observation_window, args.variant
            )
            rollout_output_path = features_path / f"{rollout_name}.pkl"

            if rollout_output_path.exists() and not args.overwrite:
                print(f"Rollout already exists at {rollout_output_path}; pass --overwrite to recompute.")
                continue

            labels_df = pd.read_parquet(
                pretraining_data / "labels" / (label_name + ".parquet")
            )
            typed_labels = [
                meds.Label(
                    subject_id=row["subject_id"],
                    prediction_time=row["prediction_time"],
                    boolean_value=row["boolean_value"],
                )
                for row in labels_df.to_dict(orient="records")
            ]
            test_labels = [l for l in typed_labels if l["subject_id"] in test_subject_ids]
            if not test_labels:
                print(f"No test-split labels for {label_name}; skipping.")
                continue

            print(
                f"Computing attention rollout for {len(test_labels)} test-set labels "
                f"(label={label_name}, top_k={args.rollout_top_k}, "
                f"tokens_per_batch={args.tokens_per_batch})"
            )
            rollout = femr.models.transformer.compute_features(
                db=database,
                model_path=str(model_dir_path),
                labels=test_labels,
                ontology=ontology,
                device=torch.device("cuda"),
                tokens_per_batch=args.tokens_per_batch,
                num_proc=args.num_proc,
                observation_window=args.observation_window,
                return_rollout=True,
                rollout_top_k=args.rollout_top_k,
            )
            with open(rollout_output_path, "wb") as f:
                pickle.dump(rollout, f)
            print(f"Rollout features saved to {rollout_output_path}")


if __name__ == "__main__":
    main()

"""End-to-end fine-tuning of a pretrained MOTOR model for a binary downstream task.

Unlike ``finetune_motor.py`` (a frozen linear probe: compute_features ->
sklearn LogisticRegressionCV), this script attaches a fresh binary
classification head to the pretrained FEMR encoder and backpropagates through
ALL transformer weights on the supervised labels.

Pipeline:
  1. Load the tokenizer (dictionary.msgpack + ontology) and the pretrained MOTOR
     checkpoint (config.json + model.safetensors).
  2. Read the label parquet (subject_id / prediction_time / boolean_value) and
     split subjects into train / val / test. ``main_split.csv`` only carries
     train/test, so a validation set is hash-carved out of train for early
     stopping.
  3. Swap the MOTOR head for a BinaryClassificationTaskHead (keeps the encoder +
     any reasoning layer weights, gives a randomly-initialized head).
  4. Build per-split batch datasets with the FEMRBatchProcessor and train with
     the HuggingFace Trainer + early stopping.
  5. Predict on the held-out test split and write metrics.json +
     test_predictions/predictions.parquet in the same schema as the linear
     probe, so results are directly comparable.

Example (single node, multi-GPU):
    USE_TORCH=1 torchrun --standalone --nnodes=1 --nproc_per_node=4 \
        -m femr.omop_meds_tutorial.finetune_motor_e2e \
        --meds_reader   /path/truveta_meds_koa_reader \
        --model_dir     /path/motor_models/plain_lr2e-4/checkpoint-163368 \
        --tokenizer_dir /path/truveta_meds_koa_motor/tokenizer \
        --ontology      /path/truveta_meds_koa_motor/ontology.pkl \
        --labels        /path/labels/tkr_since_koa.parquet \
        --main_split    /path/motor/main_split.csv \
        --output_dir    /path/results/tkr_since_koa/motor_finetune \
        --num_train_epochs 10 --learning_rate 2e-5 ... (TrainingArguments)
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
import pickle
from typing import List, Optional, Tuple

import datasets
import meds
import numpy as np
import pandas as pd
import sklearn.metrics
import torch
import transformers
from transformers import HfArgumentParser, TrainingArguments

import femr.models.config
import femr.models.processor
import femr.models.tokenizer
import femr.models.transformer
import femr.splits
import meds_reader
from femr.models.tasks import BinaryClassificationTask
from femr.models.transformer import BinaryClassificationTaskHead


class CustomEarlyStoppingCallback(transformers.EarlyStoppingCallback):
    """Early stopping on *relative* improvement of the monitored metric.

    Identical to the callback used in pretrain_motor.py so fine-tuning shares
    the same convergence semantics (patience 1, relative threshold 1e-3).
    """

    def check_metric_value(self, args, state, control, metric_value):
        operator = np.greater if args.greater_is_better else np.less
        if state.best_metric is None or (
                operator(metric_value, state.best_metric)
                and abs(metric_value - state.best_metric) / state.best_metric
                > self.early_stopping_threshold
        ):
            self.early_stopping_patience_counter = 0
        else:
            self.early_stopping_patience_counter += 1


@dataclasses.dataclass
class FinetuneArguments:
    meds_reader: str = dataclasses.field(
        metadata={"help": "Path to the meds_reader SubjectDatabase."},
    )
    model_dir: str = dataclasses.field(
        metadata={"help": "Pretrained MOTOR checkpoint dir (config.json + model.safetensors)."},
    )
    tokenizer_dir: str = dataclasses.field(
        metadata={"help": "Dir holding dictionary.msgpack for the HierarchicalTokenizer."},
    )
    ontology: str = dataclasses.field(
        metadata={"help": "Path to ontology.pkl."},
    )
    labels: str = dataclasses.field(
        metadata={"help": "Path to the label parquet (subject_id / prediction_time / boolean_value)."},
    )
    main_split: str = dataclasses.field(
        metadata={"help": "Path to main_split.csv (train/test subject split)."},
    )
    tokens_per_batch: int = dataclasses.field(
        default=16384,
        metadata={"help": "Maximum number of tokens per batch."},
    )
    num_proc: int = dataclasses.field(
        default=8,
        metadata={"help": "Number of processes used when building batches."},
    )
    val_frac: float = dataclasses.field(
        default=0.1,
        metadata={"help": "Fraction of train subjects hash-carved into a validation set for early stopping."},
    )
    val_split_seed: int = dataclasses.field(
        default=97,
        metadata={"help": "Seed for the deterministic train/val hash split."},
    )
    observation_window: Optional[int] = dataclasses.field(
        default=None,
        metadata={"help": "Optional observation window (days) for feature/label extraction."},
    )
    pos_weight: Optional[float] = dataclasses.field(
        default=None,
        metadata={"help": "Optional positive-class weight in the BCE loss (handles class imbalance)."},
    )
    freeze_encoder: bool = dataclasses.field(
        default=False,
        metadata={"help": "If True, freeze the transformer encoder and train only the classification head."},
    )
    early_stopping_patience: int = dataclasses.field(
        default=1,
        metadata={"help": "Stop after this many consecutive evals without a qualifying improvement."},
    )
    early_stopping_threshold: float = dataclasses.field(
        default=0.001,
        metadata={"help": "Minimum relative improvement (|new-best|/best) counted as an improvement."},
    )
    max_train_subjects: Optional[int] = dataclasses.field(
        default=None,
        metadata={"help": "Debug: cap the number of train subjects (smoke tests). None = all."},
    )
    max_eval_subjects: Optional[int] = dataclasses.field(
        default=None,
        metadata={"help": "Debug: cap the number of val/test subjects. None = all."},
    )


def parse_arguments() -> Tuple[FinetuneArguments, TrainingArguments]:
    parser = HfArgumentParser((FinetuneArguments, TrainingArguments))
    return parser.parse_args_into_dataclasses()


def load_labels(labels_path: str) -> pd.DataFrame:
    df = pd.read_parquet(labels_path)
    required = {"subject_id", "prediction_time", "boolean_value"}
    missing = required - set(df.columns)
    assert not missing, f"label parquet is missing columns: {missing}"
    df["prediction_time"] = pd.to_datetime(df["prediction_time"])
    return df


def labels_to_meds(df: pd.DataFrame) -> List[meds.Label]:
    return [
        meds.Label(
            subject_id=int(rec["subject_id"]),
            prediction_time=rec["prediction_time"],
            boolean_value=bool(rec["boolean_value"]),
        )
        for rec in df.to_dict(orient="records")
    ]


def build_or_load_batches(
        cache_dir: pathlib.Path,
        split_name: str,
        labels_df: pd.DataFrame,
        subject_ids: List[int],
        tokenizer,
        finetune_args: FinetuneArguments,
        training_args: TrainingArguments,
) -> datasets.Dataset:
    """Build (or load from disk) the batch dataset for one split.

    Wrapped in main_process_first so that under DDP only the main process builds
    and saves the batches; the other ranks then load the cached dataset.
    """
    split_cache = cache_dir / split_name
    with training_args.main_process_first(desc=f"build batches: {split_name}"):
        if split_cache.exists():
            print(f"[{split_name}] loading cached batches from {split_cache}", flush=True)
            return datasets.Dataset.load_from_disk(str(split_cache))

        split_ids = set(int(s) for s in subject_ids)
        split_df = labels_df[labels_df["subject_id"].isin(split_ids)]
        print(f"[{split_name}] {len(split_df)} labels over {split_df.subject_id.nunique()} subjects", flush=True)

        task = BinaryClassificationTask(
            labels_to_meds(split_df), observation_window=finetune_args.observation_window
        )
        processor = femr.models.processor.FEMRBatchProcessor(tokenizer, task)

        with meds_reader.SubjectDatabase(finetune_args.meds_reader, num_threads=finetune_args.num_proc) as db:
            filtered = db.filter(list(split_df["subject_id"].astype(int).unique()))
            batches = processor.convert_dataset(
                filtered,
                tokens_per_batch=finetune_args.tokens_per_batch,
                min_subjects_per_batch=1,
                num_proc=finetune_args.num_proc,
            )
        batches.set_format("pt")
        batches.save_to_disk(str(split_cache))
        return datasets.Dataset.load_from_disk(str(split_cache))


def load_model(finetune_args: FinetuneArguments, train_task: BinaryClassificationTask):
    """Load the pretrained MOTOR encoder and swap in a fresh classification head.

    We load with the checkpoint's own (MOTOR) task config so all encoder /
    reasoning weights load cleanly, then replace the head. Swapping after load
    sidesteps from_pretrained erroring on the differently-shaped task head.
    """
    model = femr.models.transformer.FEMRModel.from_pretrained(finetune_args.model_dir)

    hidden_size = model.config.transformer_config.hidden_size
    model.config.task_config = train_task.get_task_config()
    head_kwargs = {}
    if finetune_args.pos_weight is not None:
        head_kwargs["pos_weight"] = finetune_args.pos_weight
    model.task_model = BinaryClassificationTaskHead(hidden_size, **head_kwargs)

    if finetune_args.freeze_encoder:
        for p in model.transformer.parameters():
            p.requires_grad = False
        print("Encoder frozen: training classification head only.", flush=True)

    return model


@torch.no_grad()
def do_predict(model, tokenizer, finetune_args, test_df, output_dir: pathlib.Path):
    """Score the held-out test split and write metrics + predictions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    task = BinaryClassificationTask(
        labels_to_meds(test_df), observation_window=finetune_args.observation_window
    )
    processor = femr.models.processor.FEMRBatchProcessor(tokenizer, task)
    with meds_reader.SubjectDatabase(finetune_args.meds_reader, num_threads=finetune_args.num_proc) as db:
        filtered = db.filter(list(test_df["subject_id"].astype(int).unique()))
        batches = processor.convert_dataset(
            filtered,
            tokens_per_batch=finetune_args.tokens_per_batch,
            min_subjects_per_batch=1,
            num_proc=finetune_args.num_proc,
        )
    batches.set_format("pt")
    loader = torch.utils.data.DataLoader(
        batches, num_workers=finetune_args.num_proc, pin_memory=True, collate_fn=processor.collate
    )

    all_subject_ids: List[int] = []
    all_probs: List[float] = []
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch in loader:
            batch = femr.models.transformer.to_device(batch, device)
            _, result = model(**batch, return_logits=True)
            logits = result["logits"].float().reshape(-1).cpu().numpy()
            probs = 1.0 / (1.0 + np.exp(-logits))
            all_probs.extend(probs.tolist())
            all_subject_ids.extend(result["subject_ids"].cpu().numpy().reshape(-1).tolist())

    # One label per subject for this cohort: join predictions back to labels by subject_id.
    pred_df = pd.DataFrame({"subject_id": all_subject_ids, "predicted_boolean_probability": all_probs})
    # If a subject somehow yields multiple prediction positions, keep the last.
    pred_df = pred_df.drop_duplicates(subset="subject_id", keep="last")
    merged = test_df.merge(pred_df, on="subject_id", how="inner")

    n_missing = len(test_df) - len(merged)
    if n_missing > 0:
        print(f"Warning: {n_missing} test labels had no prediction (no features).", flush=True)

    y_true = merged["boolean_value"].astype(bool).to_numpy()
    y_prob = merged["predicted_boolean_probability"].to_numpy()

    roc_auc = float(sklearn.metrics.roc_auc_score(y_true, y_prob))
    precision, recall, _ = sklearn.metrics.precision_recall_curve(y_true, y_prob)
    pr_auc = float(sklearn.metrics.auc(recall, precision))

    pred_out_dir = output_dir / "test_predictions"
    pred_out_dir.mkdir(exist_ok=True, parents=True)
    pd.DataFrame({
        "subject_id": merged["subject_id"].to_numpy(),
        "prediction_time": merged["prediction_time"].to_numpy(),
        "predicted_boolean_probability": y_prob,
        "predicted_boolean_value": pd.array([None] * len(merged), dtype="boolean"),
        "boolean_value": y_true,
    }).to_parquet(pred_out_dir / "predictions.parquet", index=False)

    metrics = {"auroc": roc_auc, "aucpr": pr_auc, "n_test": int(len(merged))}
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Test AUROC={roc_auc:.4f}  AUCPR={pr_auc:.4f}  (n={len(merged)})", flush=True)
    return metrics


def main():
    finetune_args, training_args = parse_arguments()
    output_dir = pathlib.Path(training_args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    with open(finetune_args.ontology, "rb") as f:
        ontology = pickle.load(f)

    tokenizer = femr.models.tokenizer.HierarchicalTokenizer.from_pretrained(
        finetune_args.tokenizer_dir, ontology=ontology
    )

    labels_df = load_labels(finetune_args.labels)

    # Split subjects: main_split gives train/test; carve a val set out of train.
    split = femr.splits.SubjectSplit.load_from_csv(finetune_args.main_split)
    labeled_subjects = set(labels_df["subject_id"].astype(int))
    train_pool = [s for s in split.train_subject_ids if s in labeled_subjects]
    test_ids = [s for s in split.test_subject_ids if s in labeled_subjects]

    tv_split = femr.splits.generate_hash_split(
        train_pool, finetune_args.val_split_seed, frac_test=finetune_args.val_frac
    )
    train_ids = tv_split.train_subject_ids
    val_ids = tv_split.test_subject_ids

    # Disjointness guard.
    assert not (set(train_ids) & set(val_ids)), "train/val overlap"
    assert not (set(train_ids) & set(test_ids)), "train/test overlap"
    assert not (set(val_ids) & set(test_ids)), "val/test overlap"

    if finetune_args.max_train_subjects is not None:
        train_ids = train_ids[: finetune_args.max_train_subjects]
    if finetune_args.max_eval_subjects is not None:
        val_ids = val_ids[: finetune_args.max_eval_subjects]
        test_ids = test_ids[: finetune_args.max_eval_subjects]

    print(f"Subjects -> train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}", flush=True)

    cache_dir = output_dir / "batches"
    train_batches = build_or_load_batches(
        cache_dir, "train", labels_df, train_ids, tokenizer, finetune_args, training_args
    )
    val_batches = build_or_load_batches(
        cache_dir, "val", labels_df, val_ids, tokenizer, finetune_args, training_args
    )

    train_task = BinaryClassificationTask(
        labels_to_meds(labels_df[labels_df["subject_id"].isin(set(train_ids))]),
        observation_window=finetune_args.observation_window,
    )
    model = load_model(finetune_args, train_task)

    # collate is data-only for binary classification (cleanup is a no-op), so any
    # processor's collate works; build one bound to the tokenizer/task.
    collate_processor = femr.models.processor.FEMRBatchProcessor(tokenizer, train_task)

    trainer = transformers.Trainer(
        model=model,
        data_collator=collate_processor.collate,
        train_dataset=train_batches,
        eval_dataset=val_batches,
        args=training_args,
        callbacks=[CustomEarlyStoppingCallback(
            early_stopping_patience=finetune_args.early_stopping_patience,
            early_stopping_threshold=finetune_args.early_stopping_threshold,
        )],
    )

    train_result = trainer.train()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    trainer.save_model()

    # Predict on the test split from the (best) loaded model, main process only.
    if training_args.process_index == 0:
        test_df = labels_df[labels_df["subject_id"].isin(set(test_ids))]
        do_predict(trainer.model, tokenizer, finetune_args, test_df, output_dir)


if __name__ == "__main__":
    main()

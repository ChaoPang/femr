
import dataclasses
from typing import Optional, Tuple
from transformers import TrainingArguments, HfArgumentParser

import numpy as np
import transformers
import pathlib
import pickle
import datasets
import femr.models.tokenizer
import femr.models.processor
import femr.models.config
import femr.models.transformer
from femr.models.tokenizer.flat_tokenizer import FlatTokenizer


class CustomEarlyStoppingCallback(transformers.EarlyStoppingCallback):
    def check_metric_value(self, args, state, control, metric_value):
        # best_metric is set by code for load_best_model
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
class ClmbrArguments:
    pretraining_data: str = dataclasses.field(
        metadata={
            "help": "Pretraining data folder"
        },
    )
    meds_reader: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "help": "The folder for the meds reader"
        },
    )
    checkpoint_dir: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "help": "The checkpoint dir to restore the training from"
        }
    )
    n_layers: int = dataclasses.field(
        default=11,
        metadata={
            "help": "Number of transformer layers"
        },
    )


def parse_arguments() -> Tuple[ClmbrArguments, TrainingArguments]:
    parser = HfArgumentParser((ClmbrArguments, TrainingArguments))
    clmbr_args, training_args = parser.parse_args_into_dataclasses()
    return clmbr_args, training_args


def main():
    clmbr_args, training_args = parse_arguments()
    pretraining_data = pathlib.Path(clmbr_args.pretraining_data)

    tokenizer_path = pretraining_data / 'tokenizer'
    tokenizer = FlatTokenizer.from_pretrained(tokenizer_path)

    task_path = pretraining_data / 'clmbr_task.pkl'
    with open(task_path, 'rb') as f:
        clmbr_task = pickle.load(f)

    processor = femr.models.processor.FEMRBatchProcessor(tokenizer, clmbr_task)

    train_batches_path = pretraining_data / 'train_batches'
    train_batches = datasets.Dataset.load_from_disk(str(train_batches_path))

    val_batches_path = pretraining_data / 'val_batches'
    val_batches = datasets.Dataset.load_from_disk(str(val_batches_path))

    transformer_config = femr.models.config.FEMRTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        is_hierarchical=False,
        n_layers=clmbr_args.n_layers,
        use_normed_ages=True,
        use_bias=False,
        hidden_act='swiglu',
    )

    config = femr.models.config.FEMRModelConfig.from_transformer_task_configs(
        transformer_config,
        clmbr_task.get_task_config()
    )

    model = femr.models.transformer.FEMRModel(config)

    trainer = transformers.Trainer(
        model=model,
        data_collator=processor.collate,
        train_dataset=train_batches,
        eval_dataset=val_batches,
        args=training_args,
        callbacks=[CustomEarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.001)],
    )
    train_result = trainer.train(resume_from_checkpoint=clmbr_args.checkpoint_dir)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()

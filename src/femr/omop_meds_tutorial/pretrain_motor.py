
import dataclasses
from typing import Optional, Tuple
from transformers import TrainingArguments, HfArgumentParser

import numpy as np
import transformers
import pathlib
import torch
import sys
import femr.models.transformer
import pickle
import datasets
import femr.models.tokenizer
import femr.models.processor


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
class MotorArguments:
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
            "help": "Pretraining data folder"
        },
    )
    use_reasoning_layer: bool = dataclasses.field(
        default=False,
        metadata={
            "help": "Whether to insert a vocab-attention reasoning layer before the task head"
        },
    )
    reasoning_top_k: int = dataclasses.field(
        default=32,
        metadata={
            "help": "Number of top vocab tokens to attend over in the reasoning layer"
        },
    )
    reasoning_weight: float = dataclasses.field(
        default=1.0,
        metadata={
            "help": "Mixing weight alpha for the reasoning layer output (1.0 = full replace, 0.0 = hidden state only)"
        },
    )
    reasoning_embedding_init_path: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "help": "Optional location of a (vocab_size, hidden_size) torch tensor used to "
                    "initialize the reasoning_embedding weight. Accepts a local path, a "
                    "hf://<repo_id>/<filename> reference, or an https://huggingface.co/... "
                    "URL. Build the tensor with embed_vocab.py (PubMedBERT, etc.) to "
                    "warm-start the reasoning layer with text-derived concept embeddings."
        },
    )
    reasoning_constrain_to_history: bool = dataclasses.field(
        default=False,
        metadata={
            "help": "If True, restrict the reasoning layer's top-k vocab selection to "
                    "tokens that occur in the *same patient's* tokenized history at or "
                    "before each prediction position. Honors MOTOR's sample-packing "
                    "(multiple patients per batch) via per-position segment IDs."
        },
    )
    reasoning_embedding_freeze: bool = dataclasses.field(
        default=False,
        metadata={
            "help": "If True, freeze the ReasoningLayer.reasoning_embedding weight "
                    "(requires_grad=False) so the reasoning tokens are non-trainable."
        },
    )
    early_stopping_patience: int = dataclasses.field(
        default=1,
        metadata={
            "help": "Stop training after this many consecutive evaluations without a "
                    "qualifying improvement in eval_loss."
        },
    )
    early_stopping_threshold: float = dataclasses.field(
        default=0.001,
        metadata={
            "help": "Minimum *relative* improvement (|new - best| / best) required to "
                    "count as an improvement for early stopping. Set to 0.0 to count any "
                    "absolute improvement, no matter how small."
        },
    )


def parse_arguments()-> (
    Tuple[MotorArguments, TrainingArguments]
):
    parser = HfArgumentParser((MotorArguments, TrainingArguments))
    motor_args, training_args = parser.parse_args_into_dataclasses()
    return motor_args, training_args


def main():
    motor_args, training_args = parse_arguments()
    pretraining_data = pathlib.Path(motor_args.pretraining_data)

    ontology_path = pretraining_data / 'ontology.pkl'
    with open(ontology_path, 'rb') as f:
        ontology = pickle.load(f)

    tokenizer_path = pretraining_data / 'tokenizer'
    tokenizer = femr.models.tokenizer.HierarchicalTokenizer.from_pretrained(
        tokenizer_path, ontology=ontology
    )

    task_path = pretraining_data / 'motor_task.pkl'
    with open(task_path, 'rb') as f:
        motor_task = pickle.load(f)

    processor = femr.models.processor.FEMRBatchProcessor(tokenizer, motor_task)

    train_batches_path = pretraining_data / 'train_batches'
    train_batches = datasets.Dataset.load_from_disk(str(train_batches_path))

    val_batches_path = pretraining_data / 'val_batches'
    val_batches = datasets.Dataset.load_from_disk(str(val_batches_path))

    # Finally, given the batches, we can train CLMBR.
    # We can use huggingface's trainer to do this.
    transformer_config = femr.models.config.FEMRTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        is_hierarchical=isinstance(tokenizer, femr.models.tokenizer.HierarchicalTokenizer),
        n_layers=motor_args.n_layers,
        use_normed_ages=True,
        use_bias=False,
        hidden_act='swiglu',
        use_reasoning_layer=motor_args.use_reasoning_layer,
        reasoning_top_k=motor_args.reasoning_top_k,
        reasoning_weight=motor_args.reasoning_weight,
        reasoning_embedding_init_path=motor_args.reasoning_embedding_init_path,
        reasoning_constrain_to_history=motor_args.reasoning_constrain_to_history,
        reasoning_embedding_freeze=motor_args.reasoning_embedding_freeze,
    )

    config = femr.models.config.FEMRModelConfig.from_transformer_task_configs(
        transformer_config,
        motor_task.get_task_config()
    )

    model = femr.models.transformer.FEMRModel(config)
    # model = model.to(torch.device("cuda"))
    #
    # learning_rate = args.learning_rate
    # output_dir = 'tmp_trainer_' + sys.argv[1]
    # trainer_config = transformers.TrainingArguments(
    #     per_device_train_batch_size=args.per_device_train_batch_size,
    #     per_device_eval_batch_size=args.per_device_eval_batch_size,
    #
    #     learning_rate=learning_rate,
    #     output_dir=output_dir,
    #     remove_unused_columns=False,
    #     bf16=True,
    #
    #     weight_decay=0.1,
    #     adam_beta2=0.95,
    #
    #     report_to=["tensorboard"],
    #
    #     num_train_epochs=args.n_epochs,
    #
    #     warmup_steps=500,
    #
    #     logging_strategy='epoch',
    #     logging_steps=10,
    #
    #     save_strategy='epoch',
    #     evaluation_strategy='epoch',
    #
    #     # prediction_loss_only=True,
    #     dataloader_num_workers=12,
    #
    #     save_total_limit=10,
    #     load_best_model_at_end=True,
    #     metric_for_best_model="eval_loss",
    #     greater_is_better=False,
    # )

    trainer = transformers.Trainer(
        model=model,
        data_collator=processor.collate,
        train_dataset=train_batches,
        eval_dataset=val_batches,
        args=training_args,
        callbacks=[CustomEarlyStoppingCallback(
            early_stopping_patience=motor_args.early_stopping_patience,
            early_stopping_threshold=motor_args.early_stopping_threshold,
        )],
    )
    train_result = trainer.train(resume_from_checkpoint=motor_args.checkpoint_dir)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()

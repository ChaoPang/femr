import transformers
import pathlib
import torch
import sys
import femr.models.transformer
import pickle
import datasets
import femr.models.tokenizer
import femr.models.processor
from .generate_labels import create_omop_meds_tutorial_arg_parser

def create_arg_parser():
    arg_parser = create_omop_meds_tutorial_arg_parser()
    arg_parser.add_argument(
        "--learning_rate",
        dest="learning_rate",
        type=float,
        default=1e-4
    )
    arg_parser.add_argument(
        "--n_layers",
        dest="n_layers",
        type=int,
        default=12
    )
    arg_parser.add_argument(
        "--n_epochs",
        dest="n_epochs",
        type=int,
        default=100
    )
    arg_parser.add_argument(
        "--per_device_train_batch_size",
        dest="per_device_train_batch_size",
        type=int,
        default=1
    )
    arg_parser.add_argument(
        "--per_device_eval_batch_size",
        dest="per_device_eval_batch_size",
        type=int,
        default=1
    )
    return arg_parser

def main():
    args = create_arg_parser().parse_args()
    pretraining_data = pathlib.Path(args.pretraining_data)

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
    val_size = len(val_batches)
    val_batches = val_batches.select(range(min(val_size, 120)))

    # Finally, given the batches, we can train CLMBR.
    # We can use huggingface's trainer to do this.

    transformer_config = femr.models.config.FEMRTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        is_hierarchical=isinstance(tokenizer, femr.models.tokenizer.HierarchicalTokenizer),
        n_layers=args.n_layers,
        use_normed_ages=True,
        use_bias=False,
        hidden_act='swiglu',
    )

    config = femr.models.config.FEMRModelConfig.from_transformer_task_configs(transformer_config,
                                                                              motor_task.get_task_config())

    model = femr.models.transformer.FEMRModel(config)
    model = model.to(torch.device("cuda"))

    learning_rate = args.learning_rate

    trainer_config = transformers.TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,

        learning_rate=learning_rate,
        output_dir='tmp_trainer_' + sys.argv[1],
        remove_unused_columns=False,
        bf16=True,

        weight_decay=0.1,
        adam_beta2=0.95,

        report_to=["tensorboard"],

        num_train_epochs=args.n_epochs,

        warmup_steps=500,

        logging_strategy='steps',
        logging_steps=500,
        disable_tqdm=True,

        evaluation_strategy='steps',
        eval_steps=500,

        prediction_loss_only=True,
        dataloader_num_workers=12,

        save_total_limit=1,
        load_best_model_at_end=True,
    )

    trainer = transformers.Trainer(
        model=model,
        data_collator=processor.collate,
        train_dataset=train_batches,
        eval_dataset=val_batches,
        args=trainer_config,
    )

    trainer.train()


if __name__ == "__main__":
    main()
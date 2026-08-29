import pathlib
import meds_reader
import pickle
import femr.splits
from femr.models.tokenizer.flat_tokenizer import FlatTokenizer, train_tokenizer
import femr.models.tasks
import femr.models.processor
import pandas as pd


def main(args):
    pretraining_data_path = pathlib.Path(args.pretraining_data)
    meds_reader_path = pathlib.Path(args.meds_reader)
    subject_splits_path = meds_reader_path / "metadata/subject_splits.parquet"
    num_threads = args.num_threads

    with meds_reader.SubjectDatabase(str(meds_reader_path), num_threads=num_threads) as database:
        subject_ids = [_ for _ in database]

        subject_splits = pd.read_parquet(subject_splits_path)
        subject_splits = subject_splits[subject_splits.subject_id.isin(subject_ids)]
        train_tuning_split = subject_splits[~subject_splits.split.isin(["held_out"])].subject_id.tolist()
        test_split = subject_splits[subject_splits.split.isin(["held_out"])].subject_id.tolist()
        main_split = femr.splits.SubjectSplit(train_tuning_split, test_split)
        main_split.save_to_csv(str(pretraining_data_path / 'main_split.csv'))

        train_split = femr.splits.generate_hash_split(main_split.train_subject_ids, 17, frac_test=0.05)

        main_database = database.filter(main_split.train_subject_ids)
        train_database = main_database.filter(train_split.train_subject_ids)
        val_database = main_database.filter(train_split.test_subject_ids)

        tokenizer_path = pretraining_data_path / 'tokenizer'
        if not tokenizer_path.exists():
            print("Train tokenizer")
            tokenizer = train_tokenizer(
                main_database,
                vocab_size=1024 * 16,
            )
            tokenizer.save_pretrained(tokenizer_path)
        else:
            tokenizer = FlatTokenizer.from_pretrained(tokenizer_path)

        task_path = pretraining_data_path / 'clmbr_task.pkl'

        if not task_path.exists():
            print("Create CLMBR task")
            clmbr_task = femr.models.tasks.CLMBRTask(clmbr_vocab_size=tokenizer.vocab_size)

            with open(task_path, 'wb') as f:
                pickle.dump(clmbr_task, f)
        else:
            with open(task_path, 'rb') as f:
                clmbr_task = pickle.load(f)

        processor = femr.models.processor.FEMRBatchProcessor(tokenizer, clmbr_task)

        example_subject_id = list(train_database)[0]
        example_subject = train_database[example_subject_id]

        # We can do this one subject at a time
        print("Convert a single subject")
        example_batch = processor.collate([processor.convert_subject(example_subject, tensor_type='pt')])

        train_batches_path = pretraining_data_path / 'train_batches'

        if not train_batches_path.exists():
            print("Convert batches")
            train_batches = processor.convert_dataset(
                train_database,
                tokens_per_batch=args.tokens_per_batch,
                min_subjects_per_batch=1,
                num_proc=num_threads
            )

            print("Convert batches to pytorch")
            train_batches.set_format("pt")
            train_batches.save_to_disk(train_batches_path)

        val_batches_path = pretraining_data_path / 'val_batches'

        if not val_batches_path.exists():
            print("Convert val batches")
            val_batches = processor.convert_dataset(val_database, tokens_per_batch=args.tokens_per_batch, num_proc=num_threads)
            val_batches.set_format("pt")
            val_batches.save_to_disk(val_batches_path)


def create_omop_meds_tutorial_argparser():
    import argparse
    parser = argparse.ArgumentParser(description="Arguments for preparing CLMBR")
    parser.add_argument(
        "--pretraining_data",
        dest="pretraining_data",
        action="store",
        required=True,
    )
    parser.add_argument(
        "--meds_reader",
        dest="meds_reader",
        action="store",
        required=True,
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

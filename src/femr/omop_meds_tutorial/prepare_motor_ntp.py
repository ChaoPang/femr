import femr.ontology
import pathlib
import meds_reader
import pickle
import femr.splits
from femr.models.tokenizer.flat_tokenizer import train_tokenizer, FlatTokenizer
import femr.models.tasks
import femr.models.processor
from femr.omop_meds_tutorial.prepare_motor import create_omop_meds_tutorial_argparser
import pandas as pd


def main(args):
    pretraining_data_path = pathlib.Path(args.pretraining_data)
    meds_reader_path = pathlib.Path(args.meds_reader)
    subject_splits_path = meds_reader_path / "metadata/subject_splits.parquet"
    code_metadata_path = meds_reader_path / "metadata/codes.parquet"
    num_threads = args.num_threads

    with meds_reader.SubjectDatabase(str(meds_reader_path), num_threads=num_threads) as database:
        subject_ids = [_ for _ in database]
        ontology_path = pretraining_data_path / 'ontology.pkl'
        if not ontology_path.exists():
            print("Creating ontology")
            ontology = femr.ontology.Ontology(args.athena_path, code_metadata_path=str(code_metadata_path))
            print("Pruning the ontology")
            ontology.prune_to_dataset(
                database,
                prune_all_descriptions=True,
                remove_ontologies={'SPL', 'HemOnc', 'LOINC'}
            )

            with open(ontology_path, 'wb') as f:
                pickle.dump(ontology, f)
        else:
            with open(ontology_path, 'rb') as f:
                ontology = pickle.load(f)

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
            # Save the tokenizer to the same directory as the model
            tokenizer.save_pretrained(tokenizer_path)
        else:
            tokenizer = FlatTokenizer.from_pretrained(tokenizer_path)

        task_path = pretraining_data_path / 'clmbr_task.pkl'
        if not task_path.exists():
            # Second, we need to prefit the MOTOR model. This is necessary because piecewise exponential models are unstable without an initial fit
            clmbr_task = femr.models.tasks.CLMBRTask(clmbr_vocab_size=tokenizer.vocab_size)
            with open(task_path, 'wb') as f:
                pickle.dump(clmbr_task, f)

        else:
            with open(task_path, 'rb') as f:
                clmbr_task = pickle.load(f)

        processor = femr.models.processor.FEMRBatchProcessor(tokenizer, clmbr_task)
        train_batches_path = pretraining_data_path / 'train_batches'

        if not train_batches_path.exists():
            print("Convert batches")
            # But generally we want to convert entire datasets
            train_batches = processor.convert_dataset(
                train_database,
                tokens_per_batch=args.tokens_per_batch,
                min_subjects_per_batch=1,
                num_proc=num_threads
            )

            print("Convert batches to pytorch")
            # Convert our batches to pytorch tensors
            train_batches.set_format("pt")
            train_batches.save_to_disk(train_batches_path)

        val_batches_path = pretraining_data_path / 'val_batches'

        if not val_batches_path.exists():
            print("Convert val batches")
            val_batches = processor.convert_dataset(val_database, tokens_per_batch=args.tokens_per_batch, num_proc=num_threads)
            # Convert our batches to pytorch tensors
            val_batches.set_format("pt")
            val_batches.save_to_disk(val_batches_path)


if __name__ == "__main__":
    main(create_omop_meds_tutorial_argparser().parse_args())

import femr.transforms
import meds_reader
import femr.models.transformer
import pandas as pd
import pickle
import meds
import pathlib
import torch
from .generate_labels import create_omop_meds_tutorial_arg_parser, LABEL_NAMES

def create_arg_parser():
    args = create_omop_meds_tutorial_arg_parser()
    args.add_argument(
        "num_proc",
        type=int,
        required=False,
        default=6,
        help="Number of processes to use"
    )
    args.add_argument(
        "tokens_per_batch",
        type=int,
        required=False,
        default=32 * 1024,
        help="The number of tokens per batch to use"
    )
    return args

def main():
    args = create_arg_parser().parse_args()
    with meds_reader.SubjectDatabase(args.meds_reader, num_threads=6) as database:
        pretraining_data = pathlib.Path(args.pretraining_data)
        ontology_path = pretraining_data / 'ontology.pkl'

        with open(ontology_path, 'rb') as f:
            ontology = pickle.load(f)

        for label_name in LABEL_NAMES:
            labels = pd.read_parquet(
                pretraining_data / "labels" / (label_name + '.parquet')
            )
            typed_labels = [
                meds.Label(
                    subject_id=label["subject_id"],
                    prediction_time=label["prediction_time"],
                    boolean_value=label["boolean_value"],
                )
                for label in labels.to_dict(orient="records")
            ]
            features = femr.models.transformer.compute_features(
                db=database,
                model_path=str(pretraining_data / "motor_model"),
                labels=typed_labels,
                ontology=ontology,
                device=torch.device('cuda'),
                tokens_per_batch=args.tokens_per_batch,
                num_proc=args.num_proc,
            )
            with open(pretraining_data / "features" / (label_name + '_motor.pkl'), 'wb') as f:
                pickle.dump(features, f)


if __name__ == "__main__":
    main()

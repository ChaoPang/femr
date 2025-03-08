"""
FEMR also supports generating tabular feature representations, an important baseline for EHR modeling
"""
import os
import shutil
import meds_reader
import pandas as pd
import femr.featurizers
import pickle
import pathlib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegressionCV
import femr.splits
from .generate_labels import LABEL_NAMES, create_omop_meds_tutorial_arg_parser


def create_arg_parser():
    args = create_omop_meds_tutorial_arg_parser()
    args.add_argument(
        "--cohort_label",
        dest="cohort_label",
        default=None,
    )
    return args


def main():
    args = create_arg_parser().parse_args()
    pretraining_data = pathlib.Path(args.pretraining_data)
    models_path = pretraining_data / "models"
    if models_path.exists():
        shutil.rmtree(models_path)
    models_path.mkdir(exist_ok=True)

    labels = LABEL_NAMES
    if args.cohort_label is not None:
        label_path = models_path.parent / "labels" / (args.cohort_label + '.parquet')
        if label_path.exists():
            print(f"Using the user defined label at: {label_path}")
            labels = [args.cohort_label]
        else:
            raise RuntimeError(f"The user provided label does not exist at {label_path}")

    with meds_reader.SubjectDatabase(args.meds_reader, num_threads=6) as database:
        for label_name in labels:
            labels = pd.read_parquet(models_path.parent / "labels" / (label_name + '.parquet'))
            with open(models_path.parent / 'features' / (label_name + '_motor.pkl'), 'rb') as f:
                features = pickle.load(f)

            # Remove the labels that do not have features generated
            labels = labels[labels.subject_id.isin(features["subject_ids"])]
            labels = labels.sort_values(["subject_id", "prediction_time"])

            labeled_features = femr.featurizers.join_labels(features, labels)

            main_split = femr.splits.SubjectSplit.load_from_csv(str(pretraining_data / 'main_split.csv'))

            train_mask = np.isin(labeled_features['subject_ids'], main_split.train_subject_ids)
            test_mask = np.isin(labeled_features['subject_ids'], main_split.test_subject_ids)

            def apply_mask(values, mask):
                def apply(k, v):
                    if len(v.shape) == 1:
                        return v[mask]
                    elif len(v.shape) == 2:
                        return v[mask, :]
                    else:
                        assert False, f"Cannot handle {k} {v.shape}"

                return {k: apply(k, v) for k, v in values.items()}

            train_data = apply_mask(labeled_features, train_mask)
            test_data = apply_mask(labeled_features, test_mask)

            model = LogisticRegressionCV(scoring='roc_auc')
            model.fit(train_data['features'], train_data['boolean_values'])

            y_pred = model.predict_log_proba(test_data['features'])[:, 1]

            final_auroc = sklearn.metrics.roc_auc_score(test_data['boolean_values'], y_pred)

            print(label_name, final_auroc)


if __name__ == "__main__":
    main()

"""
FEMR also supports generating tabular feature representations, an important baseline for EHR modeling
"""

import femr.splits
import shutil
import meds_reader
import pandas as pd
import femr.featurizers
import pickle
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegressionCV
import optuna
import functools
import lightgbm as lgb
from .generate_labels import LABEL_NAMES, create_omop_meds_tutorial_arg_parser


def lightgbm_objective(trial, *, train_data, dev_data, num_trees=None):
    param = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,

        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    dtrain = lgb.Dataset(train_data['features'], label=train_data['boolean_values'])
    ddev = lgb.Dataset(dev_data['features'], label=dev_data['boolean_values'])

    if num_trees is None:
        callbacks = [lgb.early_stopping(10)]
        gbm = lgb.train(param, dtrain, num_boost_round=1000, valid_sets=(ddev,), callbacks=callbacks)
    else:
        gbm = lgb.train(param, dtrain, num_boost_round=num_trees)

    y_pred = gbm.predict(dev_data['features'], raw_score=True)

    error = -sklearn.metrics.roc_auc_score(dev_data['boolean_values'], y_pred)

    if num_trees is None:
        trial.set_user_attr("num_trees", gbm.best_iteration + 1)

    return error



def main():
    from pathlib import Path
    args = create_omop_meds_tutorial_arg_parser().parse_args()
    models_path = Path(args.pretraining_data) / "models"
    if models_path.exists():
        shutil.rmtree(str(models_path))
    models_path.mkdir(exist_ok=False)

    with meds_reader.SubjectDatabase(args.meds_reader, num_threads=6) as database:
        for label_name in LABEL_NAMES:
            labels = pd.read_parquet(models_path.parent / "labels" / (label_name + '.parquet'))
            with open(models_path.parent / 'features' / (label_name + '.pkl'), 'rb') as f:
                features = pickle.load(f)

            with open(models_path.parent / 'features' / (label_name + '_featurizer.pkl'), 'rb') as f:
                feautrizer: femr.featurizers.FeaturizerList = pickle.load(f)

            labeled_features = femr.featurizers.join_labels(features, labels)
            main_split = femr.splits.SubjectSplit.load_from_csv(str(models_path.parent / 'main_split.csv'))
            train_split = femr.splits.generate_hash_split(main_split.train_subject_ids, 17, frac_test=0.10)

            train_mask = np.isin(labeled_features['subject_ids'], train_split.train_subject_ids)
            dev_mask = np.isin(labeled_features['subject_ids'], train_split.test_subject_ids)
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
            dev_data = apply_mask(labeled_features, dev_mask)
            test_data = apply_mask(labeled_features, test_mask)

            lightgbm_study = optuna.create_study()  # Create a new study.
            lightgbm_study.optimize(functools.partial(lightgbm_objective, train_data=train_data, dev_data=dev_data),
                                    n_trials=100)  # Invoke optimization of the objective function.

            final_train_data = apply_mask(labeled_features, train_mask | dev_mask)

            final_lightgbm_auroc = lightgbm_objective(lightgbm_study.best_trial, train_data=final_train_data,
                                                      dev_data=test_data,
                                                      num_trees=lightgbm_study.best_trial.user_attrs['num_trees'])
            print(label_name)

            print('lightgbm', final_lightgbm_auroc, label_name)

            logistic_model = LogisticRegressionCV(scoring='roc_auc')
            logistic_model.fit(final_train_data['features'], final_train_data['boolean_values'])
            logistic_y_pred = logistic_model.predict_log_proba(test_data['features'])[:, 1]
            final_logistic_auroc = sklearn.metrics.roc_auc_score(test_data['boolean_values'], logistic_y_pred)

            print('logistic', final_logistic_auroc, label_name)


if __name__ == "__main__":
    main()
"""Time-to-event (survival) baseline, mirroring ``train_baseline.py``.

This is the survival analog of the binary GBM / logistic baseline. It consumes the
same FEMR tabular count features and the same subject splits, but trains on the
time-to-event labels produced by ``TKRLabeler(mode="tte")``
(``time_to_event_days`` + ``event_observed``) and evaluates with Harrell's
concordance index instead of AUROC.

Model (the survival analog of the LightGBM GBM in ``train_baseline.py``):
  * XGBoost with ``objective="survival:cox"`` -- histogram-based and sparse-native
    (like LightGBM), so it trains on the full multi-million-row cohort over the
    ~190k-dim sparse count matrix in minutes. Optuna-tuned with early stopping,
    mirroring the binary baseline; ``--no_optuna`` does a single fit instead.

Harrell's concordance index is the primary metric. We also report censored
ROC-AUC / PR-AUC at a horizon (default 5 years) and export the model's feature
importance (gain) mapped back to the count-feature names.

Run (mirrors train_baseline.py args):
    python -m femr.omop_meds_tutorial.train_baseline_tte \
        --pretraining_data .../motor --meds_reader .../truveta_meds_koa_reader \
        [--cohort_label tkr_tte] [--no_optuna] [--max_train_rows N]
"""

import functools
import json
import pickle
from pathlib import Path

import femr.splits
import meds_reader
import numpy as np
import optuna
import pandas as pd
import polars as pl
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sksurv.metrics import concordance_index_censored

from .generate_labels import create_omop_meds_tutorial_arg_parser
from .generate_tabular_features import get_baseline_featurizer_name, get_baseline_features_name

# Survival labels (boolean labels are handled by train_baseline.py instead).
TTE_LABEL_NAMES = ["tkr_tte"]

# Base XGBoost params shared by every fit (Cox proportional-hazards survival).
XGB_BASE_PARAMS = {
    "objective": "survival:cox",
    "eval_metric": "cox-nloglik",
    "tree_method": "hist",
    "verbosity": 0,
}

# Fixed hyperparameters used when --no_optuna is set (a single fit instead of the
# Optuna search). Sensible defaults; tune later if desired.
FIXED_XGB_PARAMS = {
    **XGB_BASE_PARAMS,
    "eta": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.5,
    "min_child_weight": 10,
    "lambda": 1.0,
}


def save_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def save_to_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def join_survival_labels(features, labels):
    """Join features to survival labels.

    Mirrors ``femr.featurizers.join_labels`` (same lexsort + two-pointer scan to
    pick, for each label, the most recent feature row at-or-before the
    prediction time), but carries the survival fields ``time_to_event_days`` and
    ``event_observed`` instead of ``boolean_value``.
    """
    indices = []
    times_to_event = []
    event_observed = []
    prediction_times = []

    order = np.lexsort((features["feature_times"], features["subject_ids"]))
    feature_index = 0

    for label in labels.itertuples(index=False):
        while (feature_index + 1) < len(order):
            next_key = (
                features['subject_ids'][order[feature_index + 1]],
                features["feature_times"][order[feature_index + 1]],
            )
            if next_key <= (label.subject_id, label.prediction_time):
                feature_index += 1
            else:
                break

        is_valid = (
            (feature_index < len(order))
            and (features["subject_ids"][order[feature_index]] == label.subject_id)
            and (features["feature_times"][order[feature_index]] <= label.prediction_time)
        )
        assert is_valid, (
            f'{feature_index} {label} {features["subject_ids"][order[feature_index]]} '
            + f'{features["feature_times"][order[feature_index]]} {len(order)}'
        )
        indices.append(order[feature_index])
        times_to_event.append(label.time_to_event_days)
        event_observed.append(label.event_observed)
        prediction_times.append(label.prediction_time)

    return {
        "times_to_event": np.asarray(times_to_event, dtype=np.float64),
        "event_observed": np.asarray(event_observed, dtype=bool),
        "subject_ids": features["subject_ids"][indices],
        "times": features["feature_times"][indices],
        "features": features["features"][indices, :],
        "prediction_times": np.asarray(prediction_times),
    }


def xgb_cox_label(data):
    """XGBoost Cox label encoding: positive time for observed events, negative
    time for right-censored subjects. Times are clipped strictly positive."""
    t = np.maximum(data["times_to_event"], 1e-3)
    return np.where(data["event_observed"], t, -t)


def to_dmatrix(data, with_label=True):
    if with_label:
        return xgb.DMatrix(data["features"], label=xgb_cox_label(data))
    return xgb.DMatrix(data["features"])


def c_index(data, risk_scores):
    """Harrell's concordance index (higher risk should mean shorter time)."""
    return concordance_index_censored(
        data["event_observed"], data["times_to_event"], risk_scores
    )[0]


def binary_metrics_at_horizon(data, risk_scores, horizon_days):
    """Translate survival predictions to binary ROC-AUC / PR-AUC at a horizon.

    Applies censoring at ``horizon_days``: a subject is
      * positive  if the event is observed at-or-before the horizon,
      * negative  if known event-free past the horizon (event OR censoring after),
      * dropped   if censored *before* the horizon (true status unknown).
    The model's continuous risk score is then scored against this binary label,
    exactly like the binary baseline's AUROC (so the numbers are comparable).
    """
    tte = data["times_to_event"]
    event = data["event_observed"]
    positive = event & (tte <= horizon_days)
    negative = tte > horizon_days  # known event-free at the horizon
    known = positive | negative

    out = {
        "horizon_days": int(horizon_days),
        "n_eval": int(known.sum()),
        "n_positive": int(positive.sum()),
        "n_dropped_censored": int((~known).sum()),
        "roc_auc": None,
        "pr_auc": None,
    }
    y_true = positive[known].astype(int)
    if known.sum() > 0 and 0 < y_true.sum() < len(y_true):
        scores = risk_scores[known]
        out["roc_auc"] = float(roc_auc_score(y_true, scores))
        out["pr_auc"] = float(average_precision_score(y_true, scores))
    return out


def survival_xgb_objective(trial, *, dtrain, ddev, dev_data, num_boost_round=None):
    """Optuna objective for the XGBoost Cox model. Mirrors ``lightgbm_objective``
    (early stopping on dev) but optimizes the concordance index (negated, since
    Optuna minimizes)."""
    param = {
        **XGB_BASE_PARAMS,
        "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
        "eta": trial.suggest_float("eta", 1e-3, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 100),
    }
    if num_boost_round is None:
        bst = xgb.train(param, dtrain, num_boost_round=1000, evals=[(ddev, "dev")],
                        early_stopping_rounds=10, verbose_eval=False)
        trial.set_user_attr("num_boost_round", bst.best_iteration + 1)
    else:
        bst = xgb.train(param, dtrain, num_boost_round=num_boost_round, verbose_eval=False)
    risk = bst.predict(ddev)
    return -c_index(dev_data, risk)


def create_arg_parser():
    args = create_omop_meds_tutorial_arg_parser()
    args.add_argument("--cohort_label", dest="cohort_label", default=None)
    args.add_argument(
        "--observation_window", dest="observation_window", type=int, default=None,
        help="The observation window for extracting features",
    )
    args.add_argument(
        "--max_train_rows", dest="max_train_rows", type=int, default=None,
        help="Cap the number of GBM training rows (GBSA is far slower than "
             "LightGBM on the full 190k-dim sparse matrix). None = use all.",
    )
    args.add_argument("--n_trials", dest="n_trials", type=int, default=10)
    args.add_argument(
        "--no_optuna", dest="no_optuna", action="store_true",
        help="Skip the Optuna search and fit a single GBM with FIXED_GBM_PARAMS "
             "(fast first result).",
    )
    args.add_argument(
        "--horizons", dest="horizons", default="1825",
        help="Comma-separated horizons (days) at which to also report censored "
             "ROC-AUC / PR-AUC. Default 1825 (5 years, matching the binary baseline).",
    )
    return args


def _apply_mask(values, mask):
    def apply(k, v):
        if len(v.shape) == 1:
            return v[mask]
        elif len(v.shape) == 2:
            return v[mask, :]
        else:
            assert False, f"Cannot handle {k} {v.shape}"
    return {k: apply(k, v) for k, v in values.items()}


def _subsample(data, n):
    """Random row subsample (for capping GBM training size)."""
    total = data["features"].shape[0]
    if n is None or total <= n:
        return data
    rng = np.random.RandomState(0)
    sel = np.sort(rng.choice(total, size=n, replace=False))
    mask = np.zeros(total, dtype=bool)
    mask[sel] = True
    return _apply_mask(data, mask)


def main():
    args = create_arg_parser().parse_args()
    models_path = Path(args.pretraining_data) / "models"
    models_path.mkdir(exist_ok=True)

    labels = TTE_LABEL_NAMES
    if args.cohort_label is not None:
        label_path = models_path.parent / "labels" / (args.cohort_label + '.parquet')
        if label_path.exists():
            print(f"Using the user defined label at: {label_path}")
            labels = [args.cohort_label]
        else:
            raise RuntimeError(f"The user provided label does not exist at {label_path}")

    horizons = [int(x) for x in str(args.horizons).split(",") if x.strip()]

    output_dir = models_path.parent / "results"
    output_dir.mkdir(exist_ok=True, parents=True)
    with meds_reader.SubjectDatabase(args.meds_reader, num_threads=6) as database:  # noqa: F841
        for label_name in labels:
            if args.observation_window:
                label_output_dir = output_dir / label_name / f"baseline_tte_{args.observation_window}"
            else:
                label_output_dir = output_dir / label_name / "baseline_tte"

            label_output_dir.mkdir(exist_ok=True, parents=True)
            done_file = label_output_dir / "done"
            if done_file.exists():
                print(f"The results for {label_name} already exist because the indicator file is present at {done_file}")
                continue

            labels_df = pd.read_parquet(models_path.parent / "labels" / (label_name + '.parquet'))
            labels_df = labels_df.sort_values(["subject_id", "prediction_time"])
            labels_df = labels_df.sample(n=len(labels_df), random_state=42, replace=False)
            with open(models_path.parent / 'features' / get_baseline_features_name(label_name, args.observation_window), 'rb') as f:
                features = pickle.load(f)

            # Remove the labels that do not have features generated
            labels_df = labels_df[labels_df.subject_id.isin(features["subject_ids"])]
            labels_df = labels_df.sort_values(["subject_id", "prediction_time"])

            labeled_features = join_survival_labels(features, labels_df)
            main_split = femr.splits.SubjectSplit.load_from_csv(str(models_path.parent / 'main_split.csv'))
            train_split = femr.splits.generate_hash_split(main_split.train_subject_ids, 17, frac_test=0.10)

            train_mask = np.isin(labeled_features['subject_ids'], train_split.train_subject_ids)
            dev_mask = np.isin(labeled_features['subject_ids'], train_split.test_subject_ids)
            test_mask = np.isin(labeled_features['subject_ids'], main_split.test_subject_ids)

            train_data = _apply_mask(labeled_features, train_mask)
            dev_data = _apply_mask(labeled_features, dev_mask)
            test_data = _apply_mask(labeled_features, test_mask)

            n_ev = lambda d: int(d["event_observed"].sum())
            print(f"{label_name}: train={train_data['features'].shape[0]} (events {n_ev(train_data)}), "
                  f"dev={dev_data['features'].shape[0]} (events {n_ev(dev_data)}), "
                  f"test={test_data['features'].shape[0]} (events {n_ev(test_data)})")

            # ------------------------------------------------------------------ #
            # Survival GBM: XGBoost survival:cox (histogram, sparse-native) --
            # the GBM analog of the binary baseline's LightGBM, scaled to the
            # full cohort.
            # ------------------------------------------------------------------ #
            gbm_output_dir = label_output_dir / "gbm_survival"
            gbm_output_dir.mkdir(exist_ok=True, parents=True)
            gbm_metrics_output_file = gbm_output_dir / 'metrics.json'
            gbm_model_file = gbm_output_dir / 'model.json'
            gbm_best_params_file = gbm_output_dir / 'best_params.json'
            if (gbm_metrics_output_file.exists()
                    and gbm_model_file.exists()
                    and gbm_best_params_file.exists()):
                print(f"The result already exists for survival GBM {label_name}, it will be skipped!")
            else:
                try:
                    # Optional row cap (XGBoost scales to the full cohort, so this
                    # is off by default).
                    if args.max_train_rows:
                        train_data = _subsample(train_data, args.max_train_rows)
                    final_train_data = _apply_mask(labeled_features, train_mask | dev_mask)
                    if args.max_train_rows:
                        final_train_data = _subsample(final_train_data, args.max_train_rows)

                    dtrain = to_dmatrix(train_data)
                    ddev = to_dmatrix(dev_data)

                    if args.no_optuna:
                        best_params = dict(FIXED_XGB_PARAMS)
                        study = None
                        print(f"Skipping Optuna; single XGBoost cox fit with {best_params}")
                        bst = xgb.train(best_params, dtrain, num_boost_round=2000,
                                        evals=[(ddev, "dev")], early_stopping_rounds=20, verbose_eval=False)
                        best_num_boost_round = bst.best_iteration + 1
                    else:
                        study = optuna.create_study()
                        study.optimize(
                            functools.partial(survival_xgb_objective, dtrain=dtrain, ddev=ddev, dev_data=dev_data),
                            n_trials=args.n_trials,
                        )
                        best_params = {**XGB_BASE_PARAMS, **study.best_trial.params}
                        best_num_boost_round = study.best_trial.user_attrs["num_boost_round"]

                    print(f"Fitting final XGBoost cox: {final_train_data['features'].shape[0]} rows, "
                          f"{best_num_boost_round} rounds")
                    dfinal = to_dmatrix(final_train_data)
                    dtest = to_dmatrix(test_data, with_label=False)
                    gbm_final = xgb.train(best_params, dfinal, num_boost_round=best_num_boost_round, verbose_eval=False)

                    risk = gbm_final.predict(dtest)
                    gbm_cindex = c_index(test_data, risk)
                    gbm_horizon_metrics = [binary_metrics_at_horizon(test_data, risk, h) for h in horizons]
                    print('xgb_survival_cox', gbm_cindex, label_name)
                    for hm in gbm_horizon_metrics:
                        print(f"  @{hm['horizon_days']}d  roc_auc={hm['roc_auc']}  pr_auc={hm['pr_auc']}  "
                              f"(n={hm['n_eval']}, pos={hm['n_positive']}, dropped={hm['n_dropped_censored']})")

                    save_to_json(
                        {"label_name": label_name, "model": "xgb_survival_cox",
                         "c_index": float(gbm_cindex), "binary_at_horizon": gbm_horizon_metrics},
                        gbm_metrics_output_file,
                    )
                    gbm_final.save_model(str(gbm_model_file))
                    save_to_json(
                        {"best_params": best_params, "best_num_boost_round": int(best_num_boost_round),
                         "tuned": study is not None,
                         "best_trial_value": (float(study.best_trial.value) if study is not None else None)},
                        gbm_best_params_file,
                    )
                    gbm_predictions = pl.DataFrame({
                        "subject_id": test_data["subject_ids"].tolist(),
                        "prediction_time": test_data["prediction_times"].tolist(),
                        "predicted_risk": risk.tolist(),
                        "time_to_event_days": test_data["times_to_event"].tolist(),
                        "event_observed": test_data["event_observed"].astype(bool).tolist(),
                    })
                    gbm_test_predictions = gbm_output_dir / "test_predictions"
                    gbm_test_predictions.mkdir(exist_ok=True, parents=True)
                    gbm_predictions.write_parquet(gbm_test_predictions / "test_gbm_predictions.parquet")

                    # Feature importance (gain) mapped back to count-feature names.
                    try:
                        with open(models_path.parent / 'features'
                                  / get_baseline_featurizer_name(label_name, args.observation_window), 'rb') as ff:
                            featurizer = pickle.load(ff)
                        gain = gbm_final.get_score(importance_type="gain")
                        imp_rows = []
                        for key, val in gain.items():
                            idx = int(key[1:])  # 'f123' -> 123
                            try:
                                name = featurizer.get_column_name(idx)
                            except Exception:
                                name = key
                            imp_rows.append({"feature_index": idx, "feature": name, "gain": float(val)})
                        importance = pl.DataFrame(imp_rows).sort("gain", descending=True)
                        importance.write_parquet(gbm_output_dir / "feature_importance.parquet")
                        print("top features by gain:")
                        for r in importance.head(15).iter_rows(named=True):
                            print(f"  {r['gain']:.1f}  {r['feature']}")
                    except Exception as imp_err:
                        print(f"feature importance failed: {imp_err}")

                    if study is not None:
                        try:
                            study.trials_dataframe().to_parquet(gbm_output_dir / "optuna_trials.parquet")
                        except Exception as trials_err:
                            print(f"could not write optuna_trials.parquet: {trials_err}")
                        save_to_pickle(study, gbm_output_dir / "optuna_study.pkl")
                except Exception as e:
                    print(e)

            try:
                f = open(done_file, "x")
            except FileExistsError:
                print("File already exists.")
            finally:
                f.close()


if __name__ == "__main__":
    main()

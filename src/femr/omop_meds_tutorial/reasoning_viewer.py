"""Streamlit viewer for MOTOR-reasoning attributions.

Shows per-prediction:
  1. subject_id
  2. ground-truth label
  3. MOTOR+reasoning predicted probability
  4. reasoning tokens sorted by weight (descending), with vocab + description

Usage:

    pip install "femr[viz]"          # or just: pip install streamlit altair
    streamlit run \\
        $(python -c "import femr.omop_meds_tutorial.reasoning_viewer as m; print(m.__file__)") \\
        -- --parquet /path/to/reasoning_predictions_long.parquet

The expected parquet is the long-format extract produced by
``python -m femr.omop_meds_tutorial.extract_reasoning_predictions``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import altair as alt
import polars as pl
import streamlit as st


def _cli_args() -> Path:
    # Streamlit forwards anything after `--` to the script.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet",
        required=True,
        help="Path to the long-format reasoning-predictions parquet produced by "
             "femr.omop_meds_tutorial.extract_reasoning_predictions.",
    )
    args, _ = parser.parse_known_args()
    return Path(args.parquet)


@st.cache_data
def load(parquet_path: str) -> pl.DataFrame:
    df = pl.read_parquet(parquet_path)
    return df.sort(["subject_id", "rank"])


def main() -> None:
    st.set_page_config(page_title="MOTOR reasoning viewer", layout="wide")
    parquet_path = _cli_args()
    if not parquet_path.exists():
        st.error(f"parquet not found: {parquet_path}")
        st.stop()

    df = load(str(parquet_path))
    n_subjects = df.select(pl.col("subject_id").n_unique()).item()
    n_test = df.filter(pl.col("in_test_set"))["subject_id"].n_unique()
    st.title("MOTOR reasoning-layer viewer")
    st.caption(
        f"source: `{parquet_path.name}`  ·  {n_subjects:,} subjects "
        f"({n_test:,} in held-out test set)  ·  "
        f"top-{df['rank'].max()} reasoning tokens per subject"
    )

    # ---- sidebar filters ----
    st.sidebar.header("Filter cohort")
    only_test = st.sidebar.checkbox("Only show test-set subjects (with predicted_prob)", value=True)
    label_filter = st.sidebar.radio(
        "Ground truth",
        options=["All", "TRUE (TKR within 5y)", "FALSE (no TKR)"],
        index=0,
    )

    # predicted_prob filter only meaningful when test subjects
    test_only = df.filter(pl.col("in_test_set")) if only_test else df
    p_min = float(test_only["predicted_prob"].min() or 0.0)
    p_max = float(test_only["predicted_prob"].max() or 1.0)
    prob_range = st.sidebar.slider(
        "Predicted probability range",
        min_value=0.0, max_value=1.0,
        value=(0.0, 1.0), step=0.01,
        disabled=not only_test,
    )
    subject_search = st.sidebar.text_input("Search subject_id (exact)", value="")

    # ---- apply filters ----
    _summary_cols = ["subject_id", "prediction_time", "true_label", "predicted_prob", "in_test_set"]
    for _c in ("outcome_time", "days_to_outcome"):
        if _c in df.columns:
            _summary_cols.append(_c)
    summary = (
        df.filter(pl.col("rank") == 1)  # one row per subject
        .select(_summary_cols)
    )
    if only_test:
        summary = summary.filter(pl.col("in_test_set"))
    if label_filter == "TRUE (TKR within 5y)":
        summary = summary.filter(pl.col("true_label"))
    elif label_filter == "FALSE (no TKR)":
        summary = summary.filter(~pl.col("true_label"))
    if only_test:
        summary = summary.filter(
            pl.col("predicted_prob").is_between(prob_range[0], prob_range[1])
        )
    if subject_search.strip():
        try:
            sid = int(subject_search.strip())
            summary = summary.filter(pl.col("subject_id") == sid)
        except ValueError:
            st.sidebar.warning("subject_id must be an integer")

    n_match = summary.height
    st.sidebar.markdown(f"**{n_match:,} subjects** match filters.")

    if n_match == 0:
        st.warning("No subjects match the current filters.")
        st.stop()

    # ---- cohort table at the top ----
    st.subheader("Cohort summary")
    has_outcome = "outcome_time" in summary.columns
    cols = ["subject_id", "prediction_time", "ground_truth", "predicted_prob", "in_test_set"]
    if has_outcome:
        cols += ["outcome_time", "days_to_outcome"]
    st.dataframe(
        summary.with_columns([
            pl.col("true_label").alias("ground_truth"),
            pl.col("predicted_prob").round(4).alias("predicted_prob"),
        ]).select(cols).sort("predicted_prob", descending=True, nulls_last=True),
        use_container_width=True,
        height=240,
        column_config={
            "subject_id": st.column_config.NumberColumn(format="%d"),
            "prediction_time": st.column_config.DatetimeColumn(format="YYYY-MM-DD"),
            "predicted_prob": st.column_config.NumberColumn(format="%.4f"),
            "outcome_time": st.column_config.DatetimeColumn(format="YYYY-MM-DD"),
            "days_to_outcome": st.column_config.NumberColumn(format="%d"),
        },
    )

    # ---- subject selector + detail panel ----
    st.subheader("Per-subject reasoning attribution")
    sids = summary["subject_id"].to_list()
    default_idx = 0
    selected = st.selectbox(
        "Subject to inspect",
        options=sids,
        index=default_idx,
        format_func=lambda sid: (
            f"subject {sid}"
            f"  ·  "
            f"label={summary.filter(pl.col('subject_id')==sid)['true_label'][0]}"
            + (
                f"  ·  P(TKR)={float(summary.filter(pl.col('subject_id')==sid)['predicted_prob'][0]):.3f}"
                if summary.filter(pl.col('subject_id')==sid)['predicted_prob'][0] is not None
                else ""
            )
        ),
    )

    row0 = summary.filter(pl.col("subject_id") == selected).to_dicts()[0]
    cols = st.columns([1, 1, 1, 1, 1, 1]) if has_outcome else st.columns([1, 1, 1, 1])
    cols[0].metric("subject_id", f"{row0['subject_id']}")
    cols[1].metric("ground truth", "TRUE" if row0["true_label"] else "FALSE")
    cols[2].metric(
        "predicted P(TKR)",
        f"{row0['predicted_prob']:.4f}" if row0["predicted_prob"] is not None else "N/A (not test)",
    )
    cols[3].metric("prediction_time", str(row0["prediction_time"]))
    if has_outcome:
        outcome = row0.get("outcome_time")
        d2o = row0.get("days_to_outcome")
        cols[4].metric(
            "outcome_time (TKR)",
            str(outcome) if outcome is not None else "—",
        )
        cols[5].metric(
            "days to TKR",
            f"{int(d2o)}" if d2o is not None else "—",
        )

    tokens = (
        df.filter(pl.col("subject_id") == selected)
        .sort("rank")  # already sorted by weight desc when written
        .with_columns(
            (pl.col("weight").cum_sum().over(pl.lit(1))).alias("cumulative")
        )
    )
    st.markdown(f"**Reasoning tokens** (top-{tokens.height}, sorted by weight descending):")
    st.dataframe(
        tokens.select([
            "rank", "weight", "cumulative", "token_id",
            "code_string", "description",
        ]),
        use_container_width=True,
        height=500,
        column_config={
            "rank":        st.column_config.NumberColumn(format="%d"),
            "weight":      st.column_config.NumberColumn(format="%.4f"),
            "cumulative":  st.column_config.NumberColumn(format="%.4f"),
            "token_id":    st.column_config.NumberColumn(format="%d"),
        },
    )

    # Bar chart of weights
    chart = (
        alt.Chart(tokens.head(32).to_pandas())
        .mark_bar()
        .encode(
            x=alt.X("weight:Q", title="reasoning weight"),
            y=alt.Y(
                "rank:O",
                sort=alt.SortField("rank", order="ascending"),
                title="rank",
            ),
            tooltip=[
                alt.Tooltip("rank:O"),
                alt.Tooltip("weight:Q", format=".4f"),
                alt.Tooltip("token_id:N"),
                alt.Tooltip("code_string:N"),
                alt.Tooltip("description:N"),
            ],
        )
        .properties(
            height=20 * min(32, tokens.height),
            title=f"Reasoning-token weights for subject {selected}",
        )
    )
    st.altair_chart(chart, use_container_width=True)


if __name__ == "__main__":
    main()

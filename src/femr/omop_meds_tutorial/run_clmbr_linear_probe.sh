#!/bin/sh

SCRIPT_NAME=$(basename "$0")
COHORTS_DIR=""

show_help() {
    echo "Usage: $SCRIPT_NAME [OPTIONS]"
    echo
    echo "Generate CLMBR features and run linear probing on a set of cohorts."
    echo
    echo "Options:"
    echo "  -h, --help               Display this help message and exit"
    echo "  --pretraining_data       Path to CLMBR pretraining output directory (required)"
    echo "  --meds_reader            Path to MEDS reader directory (required)"
    echo "  --cohorts_dir            Path to directory containing cohort subdirectories (required)"
    echo
    echo "Example:"
    echo "  $SCRIPT_NAME --pretraining_data ~/clmbr_output --meds_reader ~/meds_reader --cohorts_dir ~/cohorts"
}

PRETRAINING_DATA_ARG=""
OMOP_MEDS_READER_ARG=""

while [ $# -gt 0 ]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --pretraining_data)
            PRETRAINING_DATA_ARG="$2"
            shift 2
            ;;
        --meds_reader)
            OMOP_MEDS_READER_ARG="$2"
            shift 2
            ;;
        --cohorts_dir)
            COHORTS_DIR="$2"
            shift 2
            ;;
        -*)
            echo "Error: Unknown option: $1" >&2
            echo "Try '$SCRIPT_NAME --help' for more information." >&2
            exit 1
            ;;
        *)
            echo "Error: Unexpected argument: $1" >&2
            echo "Try '$SCRIPT_NAME --help' for more information." >&2
            exit 1
            ;;
    esac
done

if [ -n "$PRETRAINING_DATA_ARG" ]; then
    PRETRAINING_DATA="$PRETRAINING_DATA_ARG"
fi

if [ -n "$OMOP_MEDS_READER_ARG" ]; then
    OMOP_MEDS_READER="$OMOP_MEDS_READER_ARG"
fi

if [ -z "$PRETRAINING_DATA" ] || [ -z "$OMOP_MEDS_READER" ] || [ -z "$COHORTS_DIR" ]; then
    echo "Error: --pretraining_data, --meds_reader, and --cohorts_dir are required." >&2
    echo "Try '$SCRIPT_NAME --help' for more information." >&2
    exit 1
fi

if [ ! -d "$COHORTS_DIR" ]; then
    echo "Error: Cohorts directory not found: $COHORTS_DIR" >&2
    exit 1
fi

echo "Using configuration:"
echo "  PRETRAINING_DATA: $PRETRAINING_DATA"
echo "  OMOP_MEDS_READER: $OMOP_MEDS_READER"
echo "  COHORTS_DIR:      $COHORTS_DIR"
echo

mkdir -p "$PRETRAINING_DATA/labels"

for cohort_path in "$COHORTS_DIR"/*/; do
    cohort_name=$(basename "$cohort_path")
    echo "========================================"
    echo "Processing cohort: $cohort_name"
    echo "========================================"

    echo "Step 1: Generating CLMBR features for $cohort_name..."
    python -u -m femr.omop_meds_tutorial.generate_clmbr_features \
        --pretraining_data "$PRETRAINING_DATA" \
        --meds_reader "$OMOP_MEDS_READER" \
        --cohort_dir "$cohort_path"

    if [ $? -ne 0 ]; then
        echo "Error: Feature generation failed for $cohort_name." >&2
        exit 1
    fi

    echo "Step 2: Running linear probe for $cohort_name..."
    python -u -m femr.omop_meds_tutorial.finetune_clmbr \
        --pretraining_data "$PRETRAINING_DATA" \
        --meds_reader "$OMOP_MEDS_READER" \
        --cohort_label "$cohort_name"

    if [ $? -ne 0 ]; then
        echo "Error: Linear probing failed for $cohort_name." >&2
        exit 1
    fi

    echo "Done with $cohort_name."
    echo
done

echo "All cohorts processed successfully."

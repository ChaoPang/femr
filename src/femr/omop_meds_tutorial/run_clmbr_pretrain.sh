#!/bin/sh

# Default values
SCRIPT_NAME=$(basename "$0")
NUM_THREADS=16
TOKENS_PER_BATCH=16384
N_LAYERS=11
NUM_TRAIN_EPOCHS=10
PER_DEVICE_TRAIN_BATCH_SIZE=1
PER_DEVICE_EVAL_BATCH_SIZE=1
LEARNING_RATE=1e-4
OUTPUT_DIR=""

# Function to display help
show_help() {
    echo "Usage: $SCRIPT_NAME [OPTIONS]"
    echo
    echo "Prepare data and pretrain a CLMBR model."
    echo
    echo "Options:"
    echo "  -h, --help                       Display this help message and exit"
    echo "  --pretraining_data               Path to pretraining data directory (required)"
    echo "  --meds_reader                    Path to MEDS reader directory (required)"
    echo "  --output_dir                     Directory to save model checkpoints (default: <pretraining_data>/clmbr_model)"
    echo "  --num_threads                    Number of threads for data preparation (default: $NUM_THREADS)"
    echo "  --tokens_per_batch               Tokens per batch for data preparation (default: $TOKENS_PER_BATCH)"
    echo "  --n_layers                       Number of transformer layers (default: $N_LAYERS)"
    echo "  --num_train_epochs               Number of training epochs (default: $NUM_TRAIN_EPOCHS)"
    echo "  --per_device_train_batch_size    Train batch size per device (default: $PER_DEVICE_TRAIN_BATCH_SIZE)"
    echo "  --per_device_eval_batch_size     Eval batch size per device (default: $PER_DEVICE_EVAL_BATCH_SIZE)"
    echo "  --learning_rate                  Learning rate (default: $LEARNING_RATE)"
    echo
    echo "Environment Variables:"
    echo "  PRETRAINING_DATA    Path to pretraining data (used if --pretraining_data not set)"
    echo "  OMOP_MEDS_READER    Path to MEDS reader (used if --meds_reader not set)"
    echo
    echo "Example:"
    echo "  $SCRIPT_NAME --pretraining_data ~/CLMBR --meds_reader ~/katara_resources/post_transform_meds_reader"
}

# Parse command line options
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
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num_threads)
            NUM_THREADS="$2"
            shift 2
            ;;
        --tokens_per_batch)
            TOKENS_PER_BATCH="$2"
            shift 2
            ;;
        --n_layers)
            N_LAYERS="$2"
            shift 2
            ;;
        --num_train_epochs)
            NUM_TRAIN_EPOCHS="$2"
            shift 2
            ;;
        --per_device_train_batch_size)
            PER_DEVICE_TRAIN_BATCH_SIZE="$2"
            shift 2
            ;;
        --per_device_eval_batch_size)
            PER_DEVICE_EVAL_BATCH_SIZE="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
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

# Use command line arguments if provided, otherwise fall back to environment variables
if [ -n "$PRETRAINING_DATA_ARG" ]; then
    PRETRAINING_DATA="$PRETRAINING_DATA_ARG"
fi

if [ -n "$OMOP_MEDS_READER_ARG" ]; then
    OMOP_MEDS_READER="$OMOP_MEDS_READER_ARG"
fi

# Check required variables
if [ -z "$PRETRAINING_DATA" ] || [ -z "$OMOP_MEDS_READER" ]; then
    echo "Error: PRETRAINING_DATA and OMOP_MEDS_READER are required." >&2
    echo "Set them as environment variables or use --pretraining_data and --meds_reader options." >&2
    echo "Try '$SCRIPT_NAME --help' for more information." >&2
    exit 1
fi

# Default output dir to <pretraining_data>/clmbr_model
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$PRETRAINING_DATA/clmbr_model"
fi

echo "Using configuration:"
echo "  PRETRAINING_DATA:              $PRETRAINING_DATA"
echo "  OMOP_MEDS_READER:              $OMOP_MEDS_READER"
echo "  OUTPUT_DIR:                    $OUTPUT_DIR"
echo "  NUM_THREADS:                   $NUM_THREADS"
echo "  TOKENS_PER_BATCH:              $TOKENS_PER_BATCH"
echo "  N_LAYERS:                      $N_LAYERS"
echo "  NUM_TRAIN_EPOCHS:              $NUM_TRAIN_EPOCHS"
echo "  PER_DEVICE_TRAIN_BATCH_SIZE:   $PER_DEVICE_TRAIN_BATCH_SIZE"
echo "  PER_DEVICE_EVAL_BATCH_SIZE:    $PER_DEVICE_EVAL_BATCH_SIZE"
echo "  LEARNING_RATE:                 $LEARNING_RATE"
echo

# Derived paths for incremental checks
TOKENIZER_PATH="$PRETRAINING_DATA/tokenizer"
TASK_PATH="$PRETRAINING_DATA/clmbr_task.pkl"
TRAIN_BATCHES_PATH="$PRETRAINING_DATA/train_batches"
VAL_BATCHES_PATH="$PRETRAINING_DATA/val_batches"
MODEL_CONFIG_PATH="$OUTPUT_DIR/config.json"

# Step 1: Prepare data (incremental — skips sub-steps whose outputs already exist)
if [ -d "$TOKENIZER_PATH" ] && [ -f "$TASK_PATH" ] && [ -d "$TRAIN_BATCHES_PATH" ] && [ -d "$VAL_BATCHES_PATH" ]; then
    echo "Step 1: Skipping data preparation (tokenizer, task, and batches already exist)."
else
    echo "Step 1: Preparing CLMBR data..."
    if [ -d "$TOKENIZER_PATH" ]; then
        echo "  - Tokenizer already exists, skipping."
    fi
    if [ -f "$TASK_PATH" ]; then
        echo "  - CLMBR task already exists, skipping."
    fi
    if [ -d "$TRAIN_BATCHES_PATH" ]; then
        echo "  - Train batches already exist, skipping."
    fi
    if [ -d "$VAL_BATCHES_PATH" ]; then
        echo "  - Val batches already exist, skipping."
    fi

    python -u src/femr/omop_meds_tutorial/prepare_clmbr.py \
        --pretraining_data "$PRETRAINING_DATA" \
        --meds_reader "$OMOP_MEDS_READER" \
        --num_threads "$NUM_THREADS" \
        --tokens_per_batch "$TOKENS_PER_BATCH"

    if [ $? -ne 0 ]; then
        echo "Error: Data preparation failed." >&2
        exit 1
    fi

    echo "Data preparation complete."
fi
echo

# Step 2: Pretrain CLMBR
if [ -f "$MODEL_CONFIG_PATH" ]; then
    echo "Step 2: Skipping pretraining (model already exists at $OUTPUT_DIR)."
else
    echo "Step 2: Pretraining CLMBR model..."
    python -u src/femr/omop_meds_tutorial/pretrain_clmbr.py \
        --pretraining_data "$PRETRAINING_DATA" \
        --meds_reader "$OMOP_MEDS_READER" \
        --n_layers "$N_LAYERS" \
        --output_dir "$OUTPUT_DIR" \
        --num_train_epochs "$NUM_TRAIN_EPOCHS" \
        --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
        --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
        --learning_rate "$LEARNING_RATE" \
        --remove_unused_columns False \
        --bf16 True \
        --weight_decay 0.1 \
        --adam_beta2 0.95 \
        --report_to tensorboard \
        --warmup_steps 500 \
        --logging_strategy epoch \
        --save_strategy epoch \
        --eval_strategy epoch \
        --dataloader_num_workers 12 \
        --save_total_limit 10 \
        --load_best_model_at_end True \
        --metric_for_best_model eval_loss \
        --greater_is_better False

    if [ $? -ne 0 ]; then
        echo "Error: Pretraining failed." >&2
        exit 1
    fi

    echo "Pretraining complete. Model saved to $OUTPUT_DIR"
fi

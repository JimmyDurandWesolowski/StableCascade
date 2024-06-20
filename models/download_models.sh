#!/bin/bash

# Exit immediately if a pipeline, which may consist of a single simple command,
# a list, or a compound command returns a non-zero status.
set -e

# Treat unset variables and parameters other than the special parameters ‘@’ or
# ‘*’, or array variables subscripted with ‘@’ or ‘*’, as an error when
# performing parameter expansion. An error message will be written to the
# standard error, and a non-interactive shell will exit.
set -u

# the return value of a pipeline is the value of the last (rightmost) command to
# exit with a non-zero status, or zero if all commands in the pipeline exit
# successfully.
set -o pipefail


URL_BASE=https://huggingface.co/stabilityai/StableWurst/resolve/main
SCRIPT_DIR=$(dirname "$0")

function model_download() {
    # -P . is already the default, but we want to make sure that the path is
    # the same as this script, and use the long options
    if [ -e "${SCRIPT_DIR}/$1" ]; then
	echo "$1 already exists, skipping"
	return
    fi
    wget "${URL_BASE}/$1" --directory-prefix="${SCRIPT_DIR}" \
	 --quiet --show-progress
}

# Check if at least two arguments were provided (excluding the optional first one)
if [ $# -lt 1 ]; then
    echo "Insufficient arguments provided. At least one argument is required."
    exit 1
fi

# Check for the optional "essential" argument and download the essential models if present
if [ "$1" == "essential" ]; then
        echo "Downloading Essential Models (EfficientNet, Stage A, Previewer)"
        model_download stage_a.safetensors
        model_download previewer.safetensors
        model_download effnet_encoder.safetensors
        shift # Move the arguments, $2 becomes $1, $3 becomes $2, etc.
fi

# Now, $1 is the second argument due to the potential shift above
second_argument="$1"
binary_decision="${2:-bfloat16}" # Use default or specific binary value if provided


STAGE_B="stage_b"
STAGE_C="stage_c"
DESC_STAGE_B="Stage B"
DESC_STAGE_C="Stage C"

case $second_argument in
    big-big)
	DESC_STAGE_B="Large ${DESC_STAGE_B}"
	DESC_STAGE_C="Large ${DESC_STAGE_C}"
	;;
    big-small)
	DESC_STAGE_B="Large ${DESC_STAGE_B}"
	DESC_STAGE_C="Small ${DESC_STAGE_C}"
        STAGE_C="${STAGE_C}_lite"
        ;;
    small-big)
	DESC_STAGE_B="Small ${DESC_STAGE_B}"
	DESC_STAGE_C="Large ${DESC_STAGE_C}"
        STAGE_B="${STAGE_B}_lite"
        ;;
    small-small)
	DESC_STAGE_B="Small ${DESC_STAGE_B}"
	DESC_STAGE_C="Small ${DESC_STAGE_C}"
        STAGE_B="${STAGE_B}_lite"
        STAGE_C="${STAGE_C}_lite"
	;;
    *)
        echo "Invalid second argument. Please provide a valid argument: " \
	   "big-big, big-small, small-big, or small-small."
        exit 2
        ;;
esac

if [ "$binary_decision" == "float32" ]; then
    DESC_FLOAT_TYPE=""
elif [ "$binary_decision" == "bfloat16" ]; then
    DESC_FLOAT_TYPE=" (BFloat16)"
    STAGE_B="${STAGE_B}_bf16"
    STAGE_C="${STAGE_C}_bf16"
else
    echo "Unknown float type ${binary_decision}"
    exit 1
fi

echo "Downloading ${DESC_STAGE_B} & ${DESC_STAGE_C}${DESC_FLOAT_TYPE}"
model_download "${STAGE_B}.safetensors"
model_download "${STAGE_C}.safetensors"

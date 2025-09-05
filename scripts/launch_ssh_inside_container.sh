#!/bin/bash

export HF_HOME=$SCRATCH/treetune_next/experiments/hf_home
IMAGE_NAME="treetune_verl_v2.sif"

# Read the api key from (script_dir)/../configs/.wandb_api_key.json
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_KEY_FILE="$SCRIPT_DIR/../.wandb_api_key.json"
ACCOUNT_FILE="$SCRIPT_DIR/../.wandb_account.json"

if [[ -f "$API_KEY_FILE" ]]; then
    export WANDB_API_KEY=$(awk -F'"' '/"key"/ {print $4}' "$API_KEY_FILE")
else
    echo "Warning: API key file not found at $API_KEY_FILE"
fi

if [[ -f "$ACCOUNT_FILE" ]]; then
    export WANDB_ENTITY=$(awk -F'"' '/"entity"/ {print $4}' "$ACCOUNT_FILE")
    export WANDB_PROJECT=$(awk -F'"' '/"project"/ {print $4}' "$ACCOUNT_FILE")
else
    echo "Warning: Account file not found at $ACCOUNT_FILE"
fi


unset WANDB_SERVICE

echo "Copying container to compute node..."
cp $SCRATCH/containers/$IMAGE_NAME $SLURM_TMPDIR/

echo "Running the container..."
cd "$HOME"
module load singularity
export SINGULARITY_CACHEDIR=$SLURM_TMPDIR/singularity_cache
mkdir -p "$SINGULARITY_CACHEDIR"

chmod a+x $SCRIPT_DIR/start_ssh_server.sh

node_name=$(scontrol show hostnames $SLURMD_NODENAME)

echo ""
echo "--------------------------------"
echo "Run on your laptop: "
echo "1. ssh -A -L 6322:localhost:6322 ${node_name}.server.mila.quebec"
echo "2. ssh mila-container"
echo "--------------------------------"
echo ""

module load singularity
export SINGULARITY_CACHEDIR=$SLURM_TMPDIR/singularity_cache
singularity exec --nv \
	-H "$HOME":"$HOME" \
	-B /network:/network \
	-B "$SLURM_TMPDIR":"$SLURM_TMPDIR" \
	"$SLURM_TMPDIR"/"$IMAGE_NAME" \
	$SCRIPT_DIR/start_ssh_server.sh -b

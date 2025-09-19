#!/bin/bash

echo "=========================================="
echo "    SUBSPACE ALPHASTEER FULL PIPELINE"
echo "=========================================="

# Configuration - Change these variables to test different models and settings
TRAIN_VAL_DIR=data/instructions/train_val
EMBEDDING_DIR=data/embeddings/llama3.1      # Output directory for embeddings
SUBSPACE_DIR=data/subspaces                 # Output directory for learned subspaces
STEERING_DIR=data/steering_matrices         # Output directory for steering matrices
NICKNAME=llama3.1
MODEL_NAME=llama3.1
HF_MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct  # HuggingFace model name

DEVICE=cuda:0

# Subspace learning parameters
K=8                     # Number of subspaces to learn
P_RATIO=0.25           # Subspace dimension as ratio of embedding dim
EPOCHS=100             # Training epochs
LR=1e-3               # Learning rate
EPSILON=0.1           # OT regularization parameter
ALPHA=0.1             # Variance preservation ratio (1-alpha kept as principal)
GAMMA=1e-3            # Regularization weight for delta learning

echo "Configuration:"
echo "  Model: $NICKNAME ($HF_MODEL_NAME)"
echo "  Device: $DEVICE"
echo "  Subspace params: K=$K, p_ratio=$P_RATIO, alpha=$ALPHA"
echo "  Training: epochs=$EPOCHS, lr=$LR, gamma=$GAMMA"
echo ""

# Create necessary directories
mkdir -p $EMBEDDING_DIR
mkdir -p $SUBSPACE_DIR
mkdir -p $STEERING_DIR
mkdir -p results

# Step 1: Extract embeddings
echo "=========================================="
echo "STEP 1: EXTRACTING EMBEDDINGS"
echo "=========================================="

for file in $TRAIN_VAL_DIR/*.json; do
    if [ -f "$file" ]; then
        filename=$(basename "$file" .json)
        echo "Extracting embeddings for $file"

        # Set prompt_column based on filename
        if [[ "$filename" == *"coconot"* ]]; then
            prompt_column="prompt"
        else
            prompt_column="query"
        fi

        python src/extract_embeddings.py --model_name $HF_MODEL_NAME \
                                        --input_file $file \
                                        --prompt_column "$prompt_column" \
                                        --output_file $EMBEDDING_DIR/embeds_$filename.pt \
                                        --batch_size 16 \
                                        --device $DEVICE
    else
        echo "No training data files found in $TRAIN_VAL_DIR"
        exit 1
    fi
done

echo "Embedding extraction completed!"
echo ""

# Step 2: Learn subspaces from benign embeddings
echo "=========================================="
echo "STEP 2: LEARNING SUBSPACES"
echo "=========================================="

python learn_benign_subspaces.py \
    --embedding_dir $EMBEDDING_DIR \
    --model_name $MODEL_NAME \
    --save_dir $SUBSPACE_DIR \
    --device $DEVICE \
    --K $K \
    --p_ratio $P_RATIO \
    --epochs $EPOCHS \
    --lr $LR \
    --epsilon $EPSILON \
    --alpha $ALPHA

echo "Subspace learning completed!"
echo ""

# Step 3: Calculate subspace-specific steering matrices
echo "=========================================="
echo "STEP 3: CALCULATING STEERING MATRICES"
echo "=========================================="

SUBSPACES_PATH=$SUBSPACE_DIR/learned_subspaces_${MODEL_NAME}.pkl

if [ ! -f "$SUBSPACES_PATH" ]; then
    echo "Error: Learned subspaces not found at $SUBSPACES_PATH"
    exit 1
fi

python calc_subspace_steering_matrix.py \
    --embedding_dir $EMBEDDING_DIR \
    --model_name $MODEL_NAME \
    --subspaces_path $SUBSPACES_PATH \
    --save_dir $STEERING_DIR \
    --device $DEVICE \
    --gamma $GAMMA

echo "Steering matrix calculation completed!"
echo ""

# Step 4: Generate sample config and demonstrate generation
echo "=========================================="
echo "STEP 4: PREPARING FOR GENERATION"
echo "=========================================="

STEERING_MATRICES_PATH=$STEERING_DIR/steering_matrices_${MODEL_NAME}.pkl
GENERATE_CONFIG_DIR=config/subspace_llama3.1

if [ ! -f "$STEERING_MATRICES_PATH" ]; then
    echo "Error: Steering matrices not found at $STEERING_MATRICES_PATH"
    exit 1
fi

# Create config directory and sample config
mkdir -p $GENERATE_CONFIG_DIR

SAMPLE_CONFIG=$GENERATE_CONFIG_DIR/sample_subspace.yaml
echo "Creating sample generation config: $SAMPLE_CONFIG"
cat > $SAMPLE_CONFIG << EOF
device: $DEVICE
model_name: $MODEL_NAME
steering_matrices_path: $STEERING_MATRICES_PATH
input_file: data/instructions/test/harmfulq_test.json
output_file: results/sample_subspace_responses.json
batch_size: 2
max_new_tokens: 256
prompt_column: query
strength: "0.0,1.0"
EOF

echo ""
echo "=========================================="
echo "    PIPELINE COMPLETED SUCCESSFULLY!"
echo "=========================================="
echo ""
echo "Summary of outputs:"
echo "  1. Embeddings: $EMBEDDING_DIR/"
echo "  2. Learned subspaces: $SUBSPACES_PATH"
echo "  3. Steering matrices: $STEERING_MATRICES_PATH"
echo "  4. Sample config: $SAMPLE_CONFIG"
echo ""
echo "To generate responses, run:"
echo "  python generate_subspace_response.py --config_path $SAMPLE_CONFIG"
echo ""
echo "Or use the generation script:"
echo "  bash scripts/generate_subspace.sh"
echo ""
echo "Pipeline completed in $(date)"
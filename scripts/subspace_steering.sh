#!/bin/bash

echo "=========================================="
echo "    SUBSPACE ALPHASTEER FULL PIPELINE"
echo "=========================================="

# Configuration - Change these variables to test different models and settings
TRAIN_VAL_DIR=data/instructions/train_val
EMBEDDING_DIR=data/embeddings/llama3.1      # Output directory for embeddings
SUBSPACE_DIR=data/subspaces/1K           # Output directory for learned subspaces
STEERING_DIR=data/steering_matrices/1K        # Output directory for steering matrices
# SUBSPACE_DATA_PATH=data/subspaces/1K/learned_subspaces_llama3.1.pkl
# STEERING_MATRICES_PATH=data/subspaces/1K/steering_matrices_llama3.1.pkl
# SUBSPACE_DATA_OUTPUT_PATH=data/combined_subspace/1K_subspace_data_llama3.1.pkl 
NICKNAME=llama3.1
MODEL_NAME=llama3.1
HF_MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct  # HuggingFace model name

DEVICE=cuda:0

# Subspace learning parameters
K=1                    # Number of subspaces to learn
P_RATIO=1.0           # Subspace dimension as ratio of embedding dim
EPOCHS=120            # Training epochs
LR=1e-3               # Learning rate
EPSILON=0.1           # OT regularization parameter
ALPHA=0.6             # Variance preservation ratio (1-alpha kept as principal)
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

echo "Learn $K subspaces with p_ratio = $p_ratio and alpha = $alpha from benign embeddings"

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


# Step 4: Combine learned subspaces and steering matrices into one pickle file
# echo "=========================================="
# echo "STEP 4: COMBINE SUBSPACE DATA"
# echo "=========================================="

# python combine_subspace_data.py \
#     --learned_subspaces_path $SUBSPACE_DATA_PATH \
#     --steering_matrices_path $STEERING_MATRICES_PATH \
#     --output_path $SUBSPACE_DATA_OUTPUT_PATH \
#     --model_name $MODEL_NAME 

# echo "Combining subspace data completed!"
# echo "Subspace data saved to $SUBSPACE_DATA_OUTPUT_PATH"


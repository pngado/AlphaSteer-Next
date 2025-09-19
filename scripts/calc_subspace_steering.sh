#!/bin/bash

# Configuration - Change these variables to test different models and settings
EMBEDDING_DIR=data/embeddings/llama3.1      # Input directory for embeddings
SUBSPACE_DIR=data/subspaces                 # Input directory for learned subspaces
STEERING_DIR=data/steering_matrices         # Output directory for steering matrices
NICKNAME=llama3.1
MODEL_NAME=llama3.1

DEVICE=cuda:0

# Delta learning parameters
GAMMA=1e-3             # Regularization weight for delta learning

# Calculate subspace-specific steering matrices
SUBSPACES_PATH=$SUBSPACE_DIR/learned_subspaces_${MODEL_NAME}.pkl
STEERING_SAVE_DIR=$STEERING_DIR

echo "Calculating subspace steering matrices for $NICKNAME"
echo "Using subspaces from: $SUBSPACES_PATH"
echo "Regularization gamma: $GAMMA"

python calc_subspace_steering_matrix.py \
    --embedding_dir $EMBEDDING_DIR \
    --model_name $MODEL_NAME \
    --subspaces_path $SUBSPACES_PATH \
    --save_dir $STEERING_SAVE_DIR \
    --device $DEVICE \
    --gamma $GAMMA

echo "Subspace steering matrix calculation completed for $NICKNAME"
echo "Results saved in: $STEERING_SAVE_DIR/steering_matrices_${MODEL_NAME}.pkl"
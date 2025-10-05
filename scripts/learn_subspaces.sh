#!/bin/bash

# Configuration - Change these variables to test different models and settings
EMBEDDING_DIR=data/embeddings/llama3.1  # Input directory for embeddings
SUBSPACE_DIR=data/subspaces/20250510_1811      # Output directory for learned subspaces
NICKNAME=llama3.1
MODEL_NAME=llama3.1

DEVICE=cuda:0

# Subspace learning parameters
K=3                     # Number of subspaces to learn
P_RATIO=0.75           # Subspace dimension as ratio of embedding dim
EPOCHS=200             # Training epochs
LR_G=1e-2              # Learning rate for g (dual variables)
LR_W=1e-3              # Learning rate for W (subspace bases)
EPSILON=0.1           # OT regularization parameter
ALPHA=0.1             # Variance preservation ratio (1-alpha kept as principal)

# Learn subspaces from benign embeddings
echo "Learning subspaces for $NICKNAME"
echo "Parameters: K=$K, p_ratio=$P_RATIO, epochs=$EPOCHS, lr_g=$LR_G, lr_W=$LR_W, alpha=$ALPHA"

python learn_benign_subspaces.py \
    --embedding_dir $EMBEDDING_DIR \
    --model_name $MODEL_NAME \
    --save_dir $SUBSPACE_DIR \
    --device $DEVICE \
    --K $K \
    --p_ratio $P_RATIO \
    --epochs $EPOCHS \
    --lr_g $LR_G \
    --lr_W $LR_W \
    --epsilon $EPSILON \
    --alpha $ALPHA \
    --use_wandb

echo "Subspace learning completed for $NICKNAME"
echo "Results saved in: $SUBSPACE_DIR/learned_subspaces_${MODEL_NAME}.pkl"
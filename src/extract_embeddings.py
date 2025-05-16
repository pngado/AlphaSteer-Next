from utils.embedding_utils import EmbeddingExtractor
import argparse
import logging 
import json
import pandas as pd
import os
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--prompt_column", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--layers", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extractor = EmbeddingExtractor(args.model_name)
    
    # Check if input file is JSON
    if os.path.exists(args.input_file) and args.input_file.endswith('.json'):
        try:
            with open(args.input_file, 'r') as f:
                data = json.load(f)
            prompts = [item[args.prompt_column] for item in data]
        except Exception as e:
            raise ValueError(f"Error loading JSON file {args.input_file}: {e}")
    else:
        raise ValueError(f"Input file {args.input_file} is not a valid JSON file")
    
    # Parse layers string to list of integers
    layers = [int(layer.strip()) for layer in args.layers.split(',')]
    
    # Extract embeddings
    embeddings = extractor.extract_embeddings(
        prompts=prompts,
        batch_size=args.batch_size,
        layers=layers
    )
    
    # Save embeddings to output file
    torch.save(embeddings, args.output_file)
    logger.info(f"Embeddings saved to {args.output_file}")

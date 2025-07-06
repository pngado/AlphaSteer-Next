#!/usr/bin/env python3
import argparse
import os
import logging
from alpaca_eval import evaluate

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='AlpacaEval evaluation script')
    
    parser.add_argument('--model-outputs', type=str, required=True,
                       help='Path to model outputs JSON file')
    parser.add_argument('--reference-outputs', type=str, required=True,
                       help='Path to reference outputs JSON file')
    parser.add_argument('--annotators-config', type=str, required=True,
                       help='Path to annotators configuration file')
    parser.add_argument('--name', type=str, required=True,
                       help='Name for this evaluation')
    parser.add_argument('--output-path', type=str, required=True,
                       help='Output directory for evaluation results')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.model_outputs):
        logging.error(f"Model outputs file does not exist: {args.model_outputs}")
        return
    
    if not os.path.exists(args.reference_outputs):
        logging.error(f"Reference outputs file does not exist: {args.reference_outputs}")
        return
    
    if not os.path.exists(args.annotators_config):
        logging.error(f"Annotators config file does not exist: {args.annotators_config}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)
    
    # Run evaluation
    try:
        logging.info(f"Starting evaluation: {args.name}")
        evaluate(
            model_outputs=args.model_outputs,
            reference_outputs=args.reference_outputs,
            annotators_config=args.annotators_config,
            name=args.name,
            output_path=args.output_path
        )
        logging.info("Evaluation completed successfully")
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")


import os
import argparse
import yaml
import json
import torch
import numpy as np
import pandas as pd

torch.manual_seed(42)
np.random.seed(42)
import time

from transformers import AutoTokenizer
from utils.const import AlphaSteer_MODELS_DICT, AlphaSteer_STEERING_LAYERS, Steer_MODELS_DICT, MODELS_DICT

import logging
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


from jinja2 import Template

template_jinja = """\
Please solve this problem, and put your final answer within \\boxed{}
This is the problem:
{{prompt}}
Please remember to put your final answer within \\boxed{}
"""


def load_config(config_path):
    '''
    Expected arguments in config:
        device: device to use
        model_name: model name
        steering_matrix_path: path to steering matrix
        input_file: path to input file
        output_file: path to output file
        batch_size: batch size
        max_new_tokens: maximum number of tokens to generate
        prompt_column: name of prompt column
    '''
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, help="Path to config file")
    return parser.parse_args()

    
if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config_path)
    for key, value in config.items():
        setattr(args, key, value)
    logger.info(f"args: {args}")

    if hasattr(args, "steering_matrix_path"):
        model_class, config_class, model_id = AlphaSteer_MODELS_DICT[args.model_name]
        if os.path.exists(args.steering_matrix_path):
            steering_matrix_or_vector = torch.load(args.steering_matrix_path, map_location=args.device)
            steering_matrix_or_vector = steering_matrix_or_vector.to(torch.bfloat16)
            logger.info(f"Generate with Null Space Steering")
            steering_layers = AlphaSteer_STEERING_LAYERS[args.model_name]
        else:
            raise ValueError("steering_matrix_path does not exist")
    elif hasattr(args, "steering_vector_path"):
        model_class, config_class, model_id = Steer_MODELS_DICT[args.model_name]
        if os.path.exists(args.steering_vector_path):
            steering_matrix_or_vector = torch.load(args.steering_vector_path, map_location=args.device)
            steering_matrix_or_vector = steering_matrix_or_vector.to(torch.bfloat16)
            logger.info(f"Generate with Naive Steering")
            steering_layers = [i for i in range(steering_matrix_or_vector.shape[0])]
        else:
            raise ValueError("steering_vector_path does not exist")
    else:
        model_class, config_class, model_id = MODELS_DICT[args.model_name]
        steering_matrix_or_vector = None
        steering_layers = None
        logger.info(f"Generate without Steering")
    
    config = config_class.from_pretrained(model_id)
    hidden_dim = config.hidden_size
    num_layers = config.num_hidden_layers
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    strength = [0.0] * num_layers
    
    model = model_class.from_pretrained(
        model_id,
        device_map=args.device,
        torch_dtype=torch.bfloat16
    )
    
    if steering_matrix_or_vector is not None:
        model.set_steering_parameters(
            steering_matrix_or_vector=steering_matrix_or_vector,
            strength=strength
        )
    else:
        logger.info(f"Generate without Steering")
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.pad_token = tokenizer.pad_token
    
    with open(args.input_file, "r") as f:
        prompts = json.load(f)
    logger.info(f"len(prompts):\t{len(prompts)}")
    if "gsm8k" in args.input_file or "math" in args.input_file:
        template = Template(template_jinja)
        messages = [{"role": "user", "content": template.render(prompt=prompt[args.prompt_column])} for prompt in prompts]
    else:
        messages = [{"role": "user", "content": prompt[args.prompt_column]} for prompt in prompts]

    formatted_prompts = [
        tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
        for message in messages
    ]
    
    total_batches = (len(formatted_prompts) + args.batch_size - 1) // args.batch_size
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_file = args.output_file.replace(".json", f"_{timestamp}.json") if hasattr(
        args, "output_file") else args.input_file.replace(".json", f"_{timestamp}.json")
    
    if hasattr(args, "strength"):
        const_strength_list = [float(s) for s in args.strength.split(",")]
    else:
        logger.info("No strength provided, using default strength")
        const_strength_list = [0.0]
    
    logger.info(f"const_strength_list: {const_strength_list}")
    completed_strengths = []
    
    try:
        for const_strength in const_strength_list:
            if steering_layers is not None:
                strength = [0.0] * num_layers
                for layer in steering_layers:
                    strength[layer] = const_strength
                model.set_steering_parameters(strength=strength)
                logger.info(f"strength: {strength}")

            for i in range(0, len(formatted_prompts), args.batch_size):
                batch_prompts = formatted_prompts[i:i + args.batch_size]
                batch_inputs = tokenizer(
                    batch_prompts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(args.device)

                input_lengths = [len(input_ids) for input_ids in batch_inputs["input_ids"]]
                batch_input_ids = batch_inputs["input_ids"]
                batch_attention_mask = batch_inputs["attention_mask"]

                start_time = time.time()
                # Generate with model
                batch_outputs = model.generate(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    num_return_sequences=1,
                    do_sample=False,
                    temperature=0.0,
                )
                end_time = time.time()

                # Process generation results
                for j, output in enumerate(batch_outputs):
                    generated_part = output[input_lengths[j]:]
                    response = tokenizer.decode(generated_part, skip_special_tokens=True)
                    prompts[i + j][f"response_strength:{const_strength}"] = response

                # Free GPU memory
                del batch_outputs, batch_input_ids, batch_attention_mask
                torch.cuda.empty_cache()

                # Print progress
                batch_idx = i // args.batch_size + 1
                time_per_example = (end_time - start_time) / min(args.batch_size, len(formatted_prompts) - i)
                
                logger.info(f"Processed batch {batch_idx}/{total_batches}, \
                            time taken: {end_time - start_time:.2f} seconds, \
                            time per example: {time_per_example:.2f} seconds")

            # Save results directly to output file
            with open(output_file, "w") as f:
                json.dump(prompts, f, indent=4)
            logger.info(f"this batch saved to {output_file}")
                
            completed_strengths.append(const_strength)
            
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        logger.error({
            "last_strength": const_strength,
            "last_batch": batch_idx,
            "completed_strengths": completed_strengths,
            "output_file": output_file,
        })
        with open(output_file, "w") as f:
            json.dump(prompts, f, indent=4)
        logger.info(f"progress saved to {output_file}")
        raise e
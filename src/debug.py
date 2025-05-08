
"""
This is a minimal example to make sure that the original GRPO training works 
and the small custom edits are working properly.
"""

import os
import yaml
import argparse
from peft import LoraConfig
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig

from utils.vllm_inference_utils import trlServer
from utils.custom_trainer import CustomGRPOTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="Qwen/Qwen2-0.5B-Instruct",
        help="The path to the model to use for training."
    )
    parser.add_argument(
        "--trl_vllm_port", 
        type=int, 
        default=8880,
        help="The port to use for the trl vllm server."
    )
    return parser.parse_args()


def reward_func(completions, **kwargs):
    # Dummy reward function that rewards completions with more unique letters.
    return [float(len(set(completion))) for completion in completions]


def main():

    args = parse_args()

    dataset = load_dataset("trl-lib/tldr", split="train")

    # define lora config
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=64,
        lora_alpha=256,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_rslora=True,
    )

    grpo_config_dict = yaml.safe_load(open('./configs/grpo.yaml', 'r'))
    grpo_config_dict['vllm_server_port'] = args.trl_vllm_port
    grpo_config = GRPOConfig(**grpo_config_dict)

    trainer = CustomGRPOTrainer(
        model=args.model_path,
        reward_funcs=reward_func,
        train_dataset=dataset,
        peft_config=lora_config,
        args=grpo_config,
    )

    print('\n\n-----------------------------------------------------------------------')
    print('Finished initializing custom grpo trainer. Training will start now.')
    print('-----------------------------------------------------------------------\n\n')

    trainer.train()


if __name__ == "__main__":
    main()
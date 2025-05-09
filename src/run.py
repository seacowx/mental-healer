
"""
This is a minimal example to make sure that the original GRPO training works 
and the small custom edits are working properly.
"""

import os
import random
import argparse
import yaml, json
import numpy as np

import torch
from torch.optim import AdamW

from peft import LoraConfig
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig

from utils.custom_trainer import CustomGRPOTrainer
from utils.data_utils import prepare_training_data
from utils.agent_utils import initialize_patient_agent
from utils.stepwise_lr_scheduler import StepWiseLRScheduler

from rewards.sentiment import SentimentReward
from rewards.therapist_reward import TherapistReward


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the RL training script.")
    parser.add_argument(
        '--base_model',
        type=str,
        default='Qwen/Qwen3-4B',
        help="The base model to use for the training.",
    )
    parser.add_argument(
        '--training_data_path',
        type=str,
        default='../data/situations/situations_with_initial_thought_top1.json',
        help="The path to the training data file. Default is '../data/situations/situations_with_initial_thought_top1.json'.",
    )
    parser.add_argument(
        '--grpo_config',
        type=str,
        default='./configs/grpo_config.yaml',
        help="The path to the training config file. Default is './configs/grpo.yaml'.",
    )
    parser.add_argument(
        "--lora_config",
        type=str,
        default='./configs/lora_config.yaml',
        help="The path to the LoRA config file. Default is './configs/lora_config.yaml'.",
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

    # STEP: load training data, each instance contains the following fields:
    # situation (str): the situation description
    # initial_thought (str): the initial thought of the patient
    # persona (str): the persona of the patient
    training_data = prepare_training_data(
        data_path=args.training_data_path,
    )

    raise SystemExit

    # STEP: initialize patient agent. The patient agent uses the same LLM as the therapist. 
    # patient_agent = initialize_patient_agent(
    #     patient_model=args.base_model,
    # )
    dataset = load_dataset("trl-lib/tldr", split="train")


    # STEP: load training config and lora config
    grpo_config_dict = yaml.safe_load(open(args.grpo_config, 'r'))
    lora_config_dict = yaml.safe_load(open(args.lora_config, 'r'))

    # define lora config
    lora_config = LoraConfig(
        **lora_config_dict
    )

    grpo_config_dict['vllm_server_port'] = args.trl_vllm_port
    grpo_config = GRPOConfig(**grpo_config_dict)

    trainer = CustomGRPOTrainer(
        model=args.base_model,
        reward_funcs=reward_func,
        train_dataset=dataset,
        peft_config=lora_config,
        args=grpo_config,
    )

    trainer.train()


if __name__ == "__main__":
    main()
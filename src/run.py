
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

from peft.config import LoraConfig
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig

from utils.custom_trainer import CustomGRPOTrainer
from utils.agent_utils import initialize_patient_agent
from utils.persona_utils import retrieve_augmented_persona
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
        default='Qwen/Qwen2-0.5B-Instruct',
        help="The base model to use for the training.",
    )
    parser.add_argument(
        '--training_config',
        type=str,
        default='./configs/grpo.yaml',
        help="The path to the training config file. Default is './configs/grpo.yaml'.",
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
    # situation_dict = json.load(open('../data/situations/situations_with_initial_thought_top1.json', 'r'))
    # augmented_persona_dict = retrieve_augmented_persona(situation_dict=situation_dict)

    # STEP: initialize patient agent. The patient agent uses the same LLM as the therapist. 
    # patient_agent = initialize_patient_agent(
    #     patient_model=args.base_model,
    # )

    dataset = load_dataset("trl-lib/tldr", split="train")

    # define lora config
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=64,
        lora_alpha=256,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # target_modules="all-linear",
        bias="none",
        use_rslora=True,
    )

    grpo_config_dict = yaml.safe_load(open(args.training_config, 'r'))
    grpo_config_dict['vllm_server_port'] = args.trl_vllm_port
    grpo_config = GRPOConfig(**grpo_config_dict)

    trainer = CustomGRPOTrainer(
        model=args.base_model,
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
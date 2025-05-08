import random
import os, sys
import argparse
import json, yaml
import numpy as np

import torch
from trl import GRPOConfig
from torch.optim import AdamW

from src.utils.vllm_inference_utils import vLLMServer
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
        default='qwen8',
        help="The base model to use for the training. Default is 'qwen3-8B'.",
    )
    parser.add_argument(
        '--training_config',
        type=str,
        default='./configs/grpo.yaml',
        help="The path to the training config file. Default is './configs/grpo.yaml'.",
    )
    return parser.parse_args()


def main():

    set_seed(96)
    args = parse_args()

    # STEP: load training data, each instance contains the following fields:
    # situation (str): the situation description
    # initial_thought (str): the initial thought of the patient
    # persona (str): the persona of the patient
    situation_dict = json.load(open('../data/situations/situations_with_initial_thought_top1.json', 'r'))
    augmented_persona_dict = retrieve_augmented_persona(situation_dict=situation_dict)


    # STEP: initialize patient agent. The patient agent uses the same LLM as the therapist. 
    patient_agent = initialize_patient_agent(
        patient_model=args.base_model,
    )

    # STEP: initialize GRPO trainer
    # TODO: initialize reward functions
    grpo_config = yaml.safe_load(open(args.training_config, 'r'))

    training_args = GRPOConfig(**grpo_config)

    # initialize the reward function
    therapist_reward = TherapistReward()

    trainer = CustomGRPOTrainer(
        model=args.base_model, 
        reward_funcs=...,
        training_args=training_args,
    )



if __name__ == "__main__":
    main()
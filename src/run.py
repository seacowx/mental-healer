import random
import os, sys
import argparse
import json, yaml
import numpy as np

import torch
from utils.llm_inference_utils import vLLMOffline
from utils.data_utils import prepare_training_data
from utils.custom_trainer import CustomGRPOTrainer

from agents.patient import Patient
from agents.therapist import Therapist
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
        '--n_personas',
        type=int,
        default=1,
        help="The number of personas to sample for each situation. Default is 1. n_personas greater than 1 will duplicate the situation.",
    )
    parser.add_argument(
        '--regenerate_thought',
        action='store_true',
        help="Whether to regenerate the initial thought. Default is False.",
    )
    parser.add_argument(
        '--disable_thinking_in_initial_thought',
        action='store_true',
        help="Whether to disable reasoning mode when producing initial thoughts. Default is False (enable reasoning mode).",
    )
    return parser.parse_args()


def main():

    set_seed(96)
    args = parse_args()

    llm_path_dict = yaml.safe_load(open('./configs/llm_configs.yaml', 'r'))

    prepared_data = prepare_training_data(
        n_personas=args.n_personas,
    )

    # Step: initialize LLM, load model in cuda:1, cuda:0 is used for reward model, cuda:2-3 for therapist agent
    therapist_reward = TherapistReward(
        sentiment_reward_device=torch.device('cuda:0'),
    )

    patient_device = torch.device('cuda:1')
    vllm = vLLMOffline(
        model_path=llm_path_dict[args.base_model]['path'],
        patient_device=patient_device,
    )

    patient_agent = Patient(
        vllm_client=vllm,
    )

    # Step: generate initial thoughts
    patient_agent.produce_initial_thought(
        data=prepared_data,
        disable_thinking=args.disable_thinking_in_initial_thought,
        therapist_reward=therapist_reward,
        regenerate_thought=args.regenerate_thought,
    )


if __name__ == "__main__":
    main()

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
    return parser.parse_args()


def main():

    set_seed(96)
    args = parse_args()

    llm_path_dict = yaml.safe_load(open('./configs/llm_configs.yaml', 'r'))

    # Step: initialize LLM, load model in cuda:1, cuda:0 is used for reward model, cuda:2-3 for therapist agent



if __name__ == "__main__":
    main()

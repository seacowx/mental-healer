import random
import os, sys
import argparse
import json, yaml
import numpy as np

import torch
from utils.llm_inference_utils import vLLMServer
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

    # STEP: load training data, each instance contains the following fields:
    # situation (str): the situation description
    # initial_thought (str): the initial thought of the patient
    # persona (str): the persona of the patient
    session_init_data = json.load(open('../data/situations/situations_with_initial_thought_top1.json', 'r'))

    # STEP: initialize vllm server for patient agent, host the server on cuda:0
    patient_vllm_async_client = vLLMServer(
        model_path=llm_path_dict[args.base_model]['path'],
        world_size=1,
        quantization=False,
    )
    patient_vllm_async_client.start_vllm_server(
        device_list=[0],
    )

    patient_llm = Patient()
    therapist_llm = Therapist()


if __name__ == "__main__":
    main()
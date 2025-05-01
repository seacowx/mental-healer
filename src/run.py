import random
import os, sys
import argparse
import json, yaml
import numpy as np

import torch
from vllm import LLM, SamplingParams
from utils.llm_inference import vLLMOffline
from utils.data_utils import prepare_training_data

from modules.patient import Patient
from modules.therapist_reward import TherapistReward


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


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
    return parser.parse_args()


def main():

    set_seed(96)
    args = parse_args()

    llm_path_dict = yaml.safe_load(open('./configs/llm_configs.yaml', 'r'))

    prepared_data = prepare_training_data(
        n_personas=args.n_personas,
    )

    # initialize LLM
    vllm = vLLMOffline(
        model_path=llm_path_dict[args.base_model]['path'],
        quantization=llm_path_dict[args.base_model]['quantization'],
    )

    patient_agent = Patient()

    # first, prompt patient agent to produce initial thought
    patient_agent.produce_initial_thought(
        data=prepared_data,
    )


if __name__ == "__main__":
    main()

import os, sys
import json
import random
import argparse
import numpy as np

import torch
from utils.llm_inference import vLLMServer
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
        '--n_personas',
        type=int,
        default=1,
        help="The number of personas to sample for each situation. Default is 1. n_personas greater than 1 will duplicate the situation.",
    )
    return parser.parse_args()


def main():

    set_seed(96)
    args = parse_args()

    prepared_data = prepare_training_data(
        n_personas=args.n_personas,
    )

    print(f"Prepared data: {len(prepared_data)} situations")


if __name__ == "__main__":
    main()

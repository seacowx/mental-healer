import os, sys
import json
import random
import argparse
import numpy as np

import torch
from utils.llm_inference import vLLMServer

from modules.patient import Patient
from modules.therapist_reward import TherapistReward


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def main():

    set_seed(96)

    data = json.load(
        open('../data/situations/situations.json', 'r')
    )

    for key, entry_dict in data.items():

        situation = entry_dict['situation']
        candidate_persona_profile_list = entry_dict['candidate_persona_profile_list']

        sampled_persona_profile = random.choice(candidate_persona_profile_list)

        print(f"Situation: {situation}")
        print(f"Sampled Persona Profile: {sampled_persona_profile}")
        raise SystemExit()


if __name__ == "__main__":
    main()

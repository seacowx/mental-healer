import os, sys
import json
import pandas as pd

from utils.llm_inference import vLLMServer
from therapist_reward import TherapistReward


def main():

    data = pd.read_csv('../data/situations/situations.csv')
    data = data.to_dict(orient='records')

    therapist_reward_func = TherapistReward()



if __name__ == "__main__":
    main()

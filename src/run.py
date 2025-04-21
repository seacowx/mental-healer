import os, sys
import json
import pandas as pd

from utils.llm_inference import vLLMServer
from train_therapist import TherapistTrainer


def main():

    data = pd.read_csv('../data/situations/situations.csv')
    data = data.to_dict(orient='records')

    therapost_trainer = TherapistTrainer(data=data)


if __name__ == "__main__":
    main()

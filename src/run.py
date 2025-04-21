import os, sys
import json
import pandas as pd

from utils.llm_inference import vLLMServer
from train_therapist import TherapistTrainer


def main():

    data = pd.read_csv('../data/situations/situations.csv')
    data = data.to_dict(orient='records')

    therapist_trainer = TherapistTrainer(data=data)

    temp_input_list = data[:10]

    therapist_trainer._TherapistTrainer__compute_sentiment_reward(
        input_msg_list=temp_input_list
    )


if __name__ == "__main__":
    main()

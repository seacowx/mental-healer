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


    test_data = json.load(
        open('../reward_ft/reward-sentiment/sentiment_data/reward-sentiment_test.json')
    )

    input_msg_list = [
            [{'role': 'user', 'content': ele['instruction'].strip()} for ele in test_data]
    ]
    label_list = [
        ele['label'].split('<emotion>')[-1].split('</emotion>')[0].lower().strip() for ele in test_data
    ]

    print(input_msg_list[0])
    print('\n\n')
    print(label_list)
    raise SystemExit()

    therapist_trainer._TherapistTrainer__compute_sentiment_reward(
        input_list=temp_input_list
    )


if __name__ == "__main__":
    main()

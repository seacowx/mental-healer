import os, sys
import json
import pandas as pd

from utils.llm_inference import vLLMServer
from train_therapist import TherapistTrainer


def main():

    data = pd.read_csv('../data/situations/situations.csv')
    data = data.to_dict(orient='records')

    print(data0[0])
    raise SystemExit()

    therapost_trainer = TherapistTrainer(data=data)

    temp_input_list = [
        {
            'situation': ele['situation'], 
            'thought': ele['thought'],
        }
        for ele in data[:10]
    ]


if __name__ == "__main__":
    main()

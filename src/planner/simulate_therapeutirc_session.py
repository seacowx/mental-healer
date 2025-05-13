import sys
import json
import argparse
sys.path.append('..')

from agents.patient import PatientAgent
from agents.therapist import TherapistAgent

from rewards.sentiment import SentimentReward

from utils.model_utils import load_all_models
from utils.data_utils import prepare_training_data
from utils.therapeutic_utils import TherapeuticSession


def parse_args():
    parser = argparse.ArgumentParser(description="Run the RL training script.")
    parser.add_argument(
        '--base_model',
        type=str,
        default='Qwen/Qwen3-8B',
        help="The base model to use for the simulating the therapeutic session.",
    )
    parser.add_argument(
        '--training_data_path',
        type=str,
        default='../data/situations/situations_with_initial_thought_top1.json',
        help= (
            "The path to the training data file. Default is " 
            "'../data/situations/situations_with_initial_thought_top1.json'.",
        )
    )
    return parser.parse_args()


def main():

    args = parse_args()

    _, _, data = prepare_training_data(data_path=args.training_data_path)

    for key, val in data.items():
        print(key)
        print(val)
        raise SystemExit

    # use Qwen3-32B as the base model to simulate the therapeutic session
    # TODO: change base model to Qwen3-32B
    offline_vllm_base_model = load_all_models(
        base_model_path=args.base_model,
    )

    sentiment_reward_model = SentimentReward(
        base_vllm_model=offline_vllm_base_model,
    )
    patient_agent = PatientAgent(
        base_vllm_model=offline_vllm_base_model,
    )
    therapist_agent = TherapistAgent(
        base_vllm_model=offline_vllm_base_model,
    )

    therapeutic_session = TherapeuticSession(
        therapist_agent=therapist_agent,
        patient_agent=patient_agent,
    )


if __name__ == '__main__':
    main()
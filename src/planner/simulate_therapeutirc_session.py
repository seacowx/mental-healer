import sys
import argparse
sys.path.append('..')

from rewards.sentiment import SentimentReward

from utils.model_utils import load_all_models
from utils.data_utils import prepare_training_data
from session.therapeutic_session import TherapeuticSession


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
        default='../../data/situations/situations_with_initial_thought_top1.json',
        help= (
            "The path to the training data file. Default is " 
            "'../../data/situations/situations_with_initial_thought_top1.json'.",
        )
    )
    return parser.parse_args()


def main():

    args = parse_args()

    _, _, data = prepare_training_data(
        data_path=args.training_data_path,
        matched_persona_path='../../data/AugESC/augesc_matched_persona.json',
        persona_hub_path='../../data/PersonaHub/persona_augmented.json',
    )

    # use Qwen3-8B as the base model to simulate the therapeutic session
    # TODO: change base model to Qwen3-32B
    offline_vllm_base_model = load_all_models(
        base_model_path=args.base_model,
        coping_chat_template_path='../prompts/coping_strategies.yaml',
    )

    sentiment_reward_model = SentimentReward(
        base_vllm_model=offline_vllm_base_model,
        reward_rule_path='../configs/sentiment_reward_rules.yaml',
        sentiment_mapping_path='../configs/emotion_to_sentiment.yaml',
    )

    therapeutic_session = TherapeuticSession(
        base_vllm_model=offline_vllm_base_model,
        coping_cot_templates_path='../prompts/coping_strategies.yaml',
        patient_prompt_template_path='../prompts/patient.yaml',
        coping_strategies_path='../configs/coping_strategy.yaml',
        sentiment_prompt_path='../prompts/sentiment.yaml',
    )

    therapeutic_session.batch_simulate_therapeutic_session(
        data=data,
        batch_size=1,
    )


if __name__ == '__main__':
    main()
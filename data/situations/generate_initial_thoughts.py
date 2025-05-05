# NOTE: Number of situations (n_personas = 1) with valid initial thoughts = 47,258

import os, sys
import argparse
import operator
import yaml, json
from copy import deepcopy
sys.path.append('../../src/')

import torch

from utils.llm_inference_utils import vLLMServer
from utils.data_utils import prepare_training_data
from rewards.therapist_reward import TherapistReward
from utils.thought_utils import iterative_thought_generation


def produce_initial_thought(
    data: dict,
    vllm_client: vLLMServer,
    therapist_reward: TherapistReward,
    top_k_personas: int = 1,
    disable_thinking: bool = False,
    regenerate_thought: bool = False,
) -> None:
    """
    Produce the initial thought given the agent's own utterance that describes a situation
    This process is done in batches.

    Inputs:
        data (dict): A dictionary containing the situation and persona profile
        therapist_reward (TherapistReward): The reward model for sentiment analysis
        disable_thinking (bool): Whether to disable reasoning mode when producing initial thoughts
        regenerate_thought (bool): Whether to regenerate the initial thought

    Outputs:
        initial_thought_list (list): A list of initial thoughts produced by the agent
    """

    initial_thought_template = yaml.safe_load(
        open('../../src/prompts/initial_thought.yaml')
    )

    # avoid re-generating the initial thought if it already exists
    cache_fpath = f'./situations_with_initial_thought_top{top_k_personas}.json'
    # if os.path.exists(cache_fpath) and not regenerate_thought:
    #     out_data = json.load(open(cache_fpath, 'r'))
    #     parsed_initial_thought_list = [
    #         val['initial_thought'] for val in out_data.values()
    #     ]
    #     return parsed_initial_thought_list
    initial_thought_message_list = []
    situation_list = []
    for key, val in data.items():

        cur_situation = val['situation']
        cur_persona = val['persona_profile']

        cur_prompt = deepcopy(initial_thought_template)

        system_content = cur_prompt['system'].replace('{{persona_profile}}', cur_persona)
        user_content = cur_prompt['user'].replace('{{persona_profile}}', cur_persona) \
            .replace('{{situation}}', cur_situation)

        cur_message = [
            {'role': 'system', 'content': system_content},
            {'role': 'user', 'content': user_content},
        ]

        situation_list.append(cur_situation)
        initial_thought_message_list.append(cur_message)

    initial_thought_message_list = initial_thought_message_list
    TOLERANCE = 5

    parsed_initial_thought_list = iterative_thought_generation(
        initial_thought_message_list=initial_thought_message_list,
        situation_list=situation_list,
        therapist_reward=therapist_reward,
        vllm_client=vllm_client,
        enable_thinking=operator.not_(disable_thinking),
        TOLERANCE=TOLERANCE,
    )

    out_data = {}
    num_invalid_thought = 0
    for initial_thought, (key, val) in zip(parsed_initial_thought_list, data.items()):

        # check if the initial thought is valid, invalid thoughts are represented as empty strings
        if initial_thought:
            out_data[key] = {
                'situation': val['situation'],
                'persona_profile': val['persona_profile'],
                'initial_thought': initial_thought,
            }
        else:
            num_invalid_thought += 1

    print(f"Number of invalid initial thoughts: {num_invalid_thought}")

    with open(cache_fpath, 'w') as f:
        json.dump(out_data, f, indent=4)
    

def parse_args():
    parser = argparse.ArgumentParser(description="Generate initial thoughts for the agent.")

    parser.add_argument(
        '--base_model',
        type=str,
        default='qwen32',
        help="The LLM to be used for the generating initial thoughts. Default is 'qwen3-32B'.",
    )
    parser.add_argument(
        '--n_personas',
        type=int,
        default=1,
        help=(
            "The number of personas to sample for each situation. Default is 1. "
            "n_personas greater than 1 will duplicate the situation."
        ))
    parser.add_argument(
        '--regenerate_thought',
        action='store_true',
        help="Whether to regenerate the initial thought. Default is False.",
    )
    parser.add_argument(
        '--disable_thinking_in_initial_thought',
        action='store_true',
        help=(
            "Whether to disable reasoning mode when producing initial thoughts. "
            "Default is False (enable reasoning mode)."
        ))

    return parser.parse_args()


def main():

    args = parse_args()

    prepared_data = prepare_training_data(
        data_path='./situations.json',
        n_personas=args.n_personas,
    )

    # DEBUG: truncate the data to 1000 situations
    prepared_data = dict(list(prepared_data.items())[:1000])

    # STEP: load LLMs via vLLM
    llm_path_dict = yaml.safe_load(open('../../src/configs/llm_configs.yaml', 'r'))

    therapist_reward = TherapistReward(
        sentiment_prompt_path='../../src/prompts/sentiment.yaml',
        sentiment_reward_device=torch.device('cuda:0'),
        llm_config_path='../../src/configs/llm_configs.yaml',
        sentiment_reward_rule_path = '../../src/configs/sentiment_reward_rules.yaml',
    )

    # when there are 4 GPUs, assuming running with A100-40G use cuda:2,3 for vLLM
    # otherwise, assuming running with A100-80G use cuda:1 for vLLM
    if torch.cuda.device_count() == 4:
        thought_device = [2, 3]
    else:
        thought_device = [1]

    # TODO: replace with vLLMServer
    vllm_client = vLLMServer(
        model_path=llm_path_dict[args.base_model]['path'],
        world_size=len(thought_device),
        quantization=False,
        devise_list=thought_device,
    )

    raise SystemExit()

    produce_initial_thought(
        data=prepared_data,
        vllm_client=vllm_client,
        therapist_reward=therapist_reward,
        top_k_personas=args.n_personas,
        disable_thinking=args.disable_thinking_in_initial_thought,
        regenerate_thought=args.regenerate_thought,
    )


if __name__ == "__main__":
    main()
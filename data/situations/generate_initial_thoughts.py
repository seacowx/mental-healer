# NOTE: Number of situations (n_personas = 1) with valid initial thoughts = 47,258

import asyncio
import os, sys
import argparse
import yaml, json
from copy import deepcopy
sys.path.append('../../src/')

import torch
from openai import AsyncOpenAI

from utils.vllm_inference_utils import vLLMServer
from utils.data_utils import augment_situation_with_persona
from utils.thought_utils import iterative_thought_generation, start_therapist_reward, stop_therapist_reward


def produce_initial_thought(
    data: dict,
    vllm_client: vLLMServer,
    thought_device: list = [],
    batch_num: int | None = None,
    top_k_personas: int = 1,
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
    batch_postfix = f'_batch{batch_num}' if batch_num is not None else ''
    cache_fpath = f'./situations_with_initial_thought_top{top_k_personas}{batch_postfix}.json'
    if os.path.exists(cache_fpath) and not regenerate_thought:
        return None

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
        vllm_client=vllm_client,
        thought_device=thought_device,
        batch_num=batch_num,
        top_k_personas=top_k_personas,
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

    return parser.parse_args()


async def main():

    args = parse_args()

    prepared_data = augment_situation_with_persona(
        data_path='./situations.json',
        n_personas=args.n_personas,
    )

    llm_path_dict = yaml.safe_load(open('../../src/configs/llm_configs.yaml', 'r'))

    if torch.cuda.device_count() == 4:
        thought_device = [0, 1, 2, 3]
    elif torch.cuda.device_count() == 2:
        thought_device = [0, 1]
    else:
        thought_device = []

    print('\n\nLoading LLMs for initial thought generation...\n')
    try:
        vllm_client = vLLMServer(
            model_path=llm_path_dict[args.base_model]['path'],
            world_size=torch.cuda.device_count(),
            quantization=False,
        )

        # divide data into batches
        batch_size = len(prepared_data) // args.n_personas

        if args.n_personas > 1:
            data_batches = [
                {k: prepared_data[k] for k in list(prepared_data.keys())[i:i + batch_size]}
                for i in range(0, len(prepared_data), batch_size)
            ]

            for batch_num, data_batch in enumerate(data_batches, start=1):
                produce_initial_thought(
                    data=data_batch,
                    vllm_client=vllm_client,
                    top_k_personas=args.n_personas,
                    thought_device=thought_device,
                    regenerate_thought=args.regenerate_thought,
                    batch_num=batch_num,
                )
        else:
            produce_initial_thought(
                data=prepared_data,
                vllm_client=vllm_client,
                top_k_personas=args.n_personas,
                thought_device=thought_device,
                regenerate_thought=args.regenerate_thought,
            )
    finally:
        vllm_client.kill_server()


if __name__ == "__main__":
    asyncio.run(main())
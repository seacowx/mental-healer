import sys
import asyncio
import json, yaml
from copy import deepcopy
from ast import literal_eval
from tqdm.asyncio import tqdm as atqdm

sys.path.append('../../src/')

import torch

from utils.llm_inference_utils import vLLMServer


async def main():

    llm_config = yaml.safe_load(open('../../src/configs/llm_configs.yaml', 'r'))
    model_path = llm_config['qwen32']['path']

    WORLD_SIZE = torch.cuda.device_count()
    vllm_server = vLLMServer(
        model_path=model_path,
        world_size=WORLD_SIZE,
        quantization=False,
    )
    async_client = vllm_server.start_vllm_server()

    # load in persona dictionary
    persona_dict = json.load(open('./persona.json', 'r'))

    # check which persona profiles are matched with AugESC
    matched_persona_dict = json.load(open('../AugESC/augsec_matched_persona.json', 'r'))
    matched_persona_keys = set()
    for val in matched_persona_dict.values():
        for sub_val in val:
            matched_persona_keys.add(sub_val['id'])

    # structurize only the matched persona profiles
    persona_dict = {
        key: val for key, val in persona_dict.items() if key in matched_persona_keys
    }

    # load prompt template
    prompt_template = yaml.safe_load(open('./prompts/structurize.yaml', 'r'))

    msg_list = []
    for key, val in persona_dict.items():

        cur_prompt_template = deepcopy(prompt_template)
        cur_prompt_template['user'] = cur_prompt_template['user'].replace(
            '{{persona_profile}}', val
        )        

        msg_list.append([
            {'role': 'system', 'content': cur_prompt_template['system']},
            {'role': 'user', 'content': cur_prompt_template['user']},
        ])

    try:
        semaphore = asyncio.Semaphore(50) 
        output_list = [
            async_client.process_with_semaphore(
                semaphore=semaphore,
                model='vllm-model',
                message=msg,
                temperature=0.6,
                max_tokens=4096,
                top_p=0.95,
                frequency_penalty=0.0,
                presence_penalty=1.0,) for msg in msg_list
            ]
        output_list = await atqdm.gather(*output_list)

    finally:
        vllm_server.kill_server()

    # get the original keys of the persona dictionary
    key_list = list(persona_dict.keys())
    structured_persona_dict = {}
    for idx, output_dict in enumerate(output_list):

        try:
            output_dict = output_dict.rsplit('</think>')[-1]
            output_dict = '{' + output_dict.split('{')[-1].split('}')[0] + '}'
            output_dict = literal_eval(output_dict)
        except:
            output_dict = {}

        cur_key = key_list[idx]
        structured_persona_dict[cur_key] = output_dict

    with open('./persona_structured.json', 'w') as f:
        json.dump(structured_persona_dict, f, indent=4)

if __name__ == "__main__":
    asyncio.run(main())
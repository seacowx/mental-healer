import sys
import json, yaml
from copy import deepcopy
from ast import literal_eval
sys.path.append('../../src/')

import torch

from utils.llm_inference_utils import vLLMServer


def main():

    llm_config = yaml.safe_load(open('../../src/configs/llm_configs.yaml', 'r'))
    model_path = llm_config['qwen32']['path']

    WORLD_SIZE = torch.cuda.device_count()
    vllm_client = vLLMServer(
        model_path=model_path,
        world_size=WORLD_SIZE,
        quantization=False,
    )

    # load in persona dictionary
    persona_dict = json.load(open('./persona.json', 'r'))

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

        print(msg_list[-1])
        raise SystemExit()


if __name__ == "__main__":
    main()
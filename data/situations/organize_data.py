"""
Organize the data for the AugESC dataset.
1. Load the AugESC dataset and the PersonaHub dataset.
2. For each situation in the AugESC dataset, find the top-10 most similar personas from the PersonaHub dataset.
3. Use LLM to filter persona profiles that are related to the situation.
4. Add the valid prosona profiles to the situation data.
"""

import re
import yaml, json
import pandas as pd
from jinja2 import Template

import torch
from vllm import LLM, SamplingParams

def make_prompt(
    persona_list: str,
    event_desc: str,
) -> str:

    template = Template(
        "You will be given a list of persona profiles and an event. "
        "Your task is to select persona profiles that can be the experiencer of the given event.\n\n"
        "<persona>\n{{ persona_list }}\n</persona>\n\n"
        "<event>\n{{ event_desc }}\n</event>\n\n"
        "If you feel unsure about a persona profile, regard it as invalid. Generate a list of indexes of the " 
        "valid persona profiles. Start the list with <valid_persona_indexes> and end it with </valid_persona_indexes>. "
        "Do not include any other text in your response."
    )

    return template.render(persona_list=persona_list, event_desc=event_desc.capitalize())


def parse_output(output):
    output = output.outputs[0].text

    print(output)
    raise SystemExit()

    if '<valid_persona_indexes>' in output and '</valid_persona_indexes>' in output:
        valid_indexes = output.split('<valid_persona_indexes>')[1].split('</valid_persona_indexes>')[0].strip().lower()
        valid_indexes = valid_indexes.replace('[', '').replace(']', '').replace(' ', '').strip()
        valid_indexes = [ele.strip() for ele in valid_indexes.split(',')]
        return valid_indexes
    return []


def main():

    situation_data =  json.load(
        open('../AugESC/augesc_content_filtered.json')
    )

    persona_data = json.load(
        open('../PersonaHub/persona.json')
    )

    matched_persona_data = json.load(
        open('../AugESC/augsec_matched_persona.json')
    )

    # initialize Qwen3-32B model
    model_path = (
        '/scratch/prj/charnu/seacow_hf_cache/models--Qwen--Qwen3-32B/'
        'snapshots/ba1f828c09458ab0ae83d42eaacc2cf8720c7957'
    )
    WORLD_SIZE = torch.cuda.device_count()
    vllm = LLM(
        model=model_path, 
        max_model_len=2048,
        tensor_parallel_size=WORLD_SIZE,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=8192,
        presence_penalty=1.,
    )

    prompt_list = []
    for key, val in situation_data.items():

        matched_persona_info_list = matched_persona_data[key]
        matched_persona_id_list = [
            ele['id'] for ele in matched_persona_info_list
        ]
        matched_persona_list = [
            persona_data[persona_id] for persona_id in matched_persona_id_list
        ]
        indexed_persona_list = [
            f"{idx}: {persona}" for idx, persona in enumerate(matched_persona_list)
        ]

        cur_prompt = make_prompt(
            persona_list='\n'.join(indexed_persona_list),
            event_desc=val,
        )

        prompt_list.append(cur_prompt)

    msg_list = [
        [{'role': 'user', 'content': ele}] for ele in prompt_list
    ]

    output_list = vllm.chat(
        messages=msg_list[:10],
        sampling_params=sampling_params,
        use_tqdm=True,
        chat_template_kwargs={
            "enable_thinking": True,
        },
    )

    output_list = [
        parse_output(output)
        for output in output_list
    ]

    print(output_list[0])
    raise SystemExit()

    augmented_situation_data = {}
    for key, val in situation_data.items():

        matched_persona_info_list = matched_persona_data[key]
        matched_persona_id_list = [
            ele['id'] for ele in matched_persona_info_list
        ]
        matched_persona_prob_dist = [
            ele['density'] for ele in matched_persona_info_list
        ]

        matched_persona_profile_list = [
            {'persona': persona_data[persona_id], 'density': persona_prob}
            for persona_id, persona_prob in zip(matched_persona_id_list, matched_persona_prob_dist)
        ]

        augmented_situation_data[key] = {
            'situation': val,
            'candidate_persona_profile_list': matched_persona_profile_list,
        }

    with open('./situations.json', 'w') as f:
        json.dump(augmented_situation_data, f, indent=4)


if __name__ == "__main__":
    main()

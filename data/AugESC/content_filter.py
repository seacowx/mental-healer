"""
Remove events from the AugESC dataset that are describing an event for others
WANT: I experienced something that had a negative impact on myself
REMOVE: My friend experienced something that had a negative impact on them

Original Size: 52,734
After Filtering: 47,260
"""
import json
import pandas as pd
from jinja2 import Template

import torch
from vllm import LLM, SamplingParams


def make_prompt(event_desc: str) -> str:

    template = Template(
        "Your task is to classiy the given event into two categories, "
        "'Related' and 'Unrelated' according to the following criterias:\n"
        "Related: The event is a personal experience and it had a negative impact on the user event at present.\n"
        "Unrelated: The event describes someone else's experience "
        "or it does not impact the users' emotion at present.\n\n"
        "<event>\n{{ event_desc }}\n</event>\n\n"
        "If the event is a recent and personal experience, respond with '<decision>Related</decision>'.\n"
        "If the event describes an outdated or someone else's experience, "
        "respond with '<decision>Unrelated</decision>'.\n"
        "Do not include any other text in your response."
    )

    return template.render(event_desc=event_desc.capitalize())


def parse_output(output):
    output = output.outputs[0].text
    if '<decision>' in output and '</decision>' in output:
        decision = output.split('<decision>')[1].split('</decision>')[0].strip().lower()
        return decision
    return ''


def main():

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

    data = json.load(open('./augesc_filtered.json', 'r'))

    prompt_list = [
        make_prompt(ele) for ele in data.values()
    ]
    msg_list = [
        [{'role': 'user', 'content': ele}] for ele in prompt_list
    ]

    output_list = vllm.chat(
        messages=msg_list,
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

    out_data = {}
    for output_label, (key, val) in zip(output_list, data.items()):
        if output_label == 'related':
            out_data[key] = val

    print(f"Original Size: {len(data)}")
    print(f"After Filtering: {len(out_data)}")

    with open('./augesc_content_filtered.json', 'w') as f:
        json.dump(out_data, f, indent=4)


if __name__ == "__main__":
    main()

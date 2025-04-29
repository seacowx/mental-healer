"""
Remove events from the AugESC dataset that are describing an event for others
WANT: I experienced something that had a negative impact on myself
REMOVE: My friend experienced something that had a negative impact on them

Original Size: 52,734
After Filtering:
"""
import json
import pandas as pd
from jinja2 import Template

import torch
from vllm import LLM, SamplingParams


def make_prompt(event_desc: str) -> str:

    template = Template(
        "Your task is to determine whether the following event describes a recent and personal experience or "
        "an outdated or others' experience.\n\n"
        "<event>\n{{ event_desc }}\n</event>\n\n"
        "If the event is a recent and personal experience, respond with '<decision>KEEP</decision>'.\n"
        "If the event describes an outdated or someone else's experience, respond with '<decision>REMOVE</decision>'.\n"
        "Do not include any other text in your response."
    )

    return template.render(event_desc=event_desc.capitalize())


def parse_output(output):
    output = output.outputs[0].text
    if '<decision>' in output and '</decision>' in output:
        decision = output.split('<decision>')[1].split('</decision>')[0].strip()
        return decision
    else:
        print(output)
        raise SystemExit()
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
        temperature=0,
        max_tokens=1024,
    )

    data = json.load(open('./augesc_filtered.json', 'r'))

    prompt_list = [
        make_prompt(ele) for ele in data.values()
    ]
    msg_list = [
        [{'role': 'user', 'content': ele}] for ele in prompt_list
    ]

    output_list = vllm.chat(
        messages=msg_list[:200],
        sampling_params=sampling_params,
        use_tqdm=True,
    )

    output_list = [
        parse_output(output)
        for output in output_list
    ]

    out_pd = pd.DataFrame({'event': list(data.values())[:200], 'decision': output_list})
    out_pd.to_csv(
        './temp_augesc_filtered.csv',
        index=False,
    )


if __name__ == "__main__":
    main()
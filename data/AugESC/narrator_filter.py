"""
Remove events from the AugESC dataset that are describing an event for others
WANT: I experienced something that had a negative impact on myself
REMOVE: My friend experienced something that had a negative impact on them

Original Size: 52,734
"""
import json
from jinja2 import Template

import torch
from vllm import LLM, SamplingParams


def make_prompt(event_desc: str) -> str:

    template = Template(
        "Your task is to determine whether the following event describes a personal experience or "
        "an experience of someone else.\n\n"
        "<event>\n{{ event_desc }}\n</event>\n\n"
        "If the event is a personal experience, respond with 'KEEP'.\n"
        "If the event describes someone else's experience, respond with 'REMOVE'.\n"
        "Do not include any other text in your response."
    )

    return template.render(event_desc=event_desc.capitalize())


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
        max_tokens=128,
        stop=["KEEP", "REMOVE"],
    )

    data = json.load(open('./augesc_filtered.json', 'r'))

    prompt_list = [
        make_prompt(ele) for ele in data.values()
    ]

    output_list = vllm.chat(
        messages=prompt_list,
        sampling_params=sampling_params,
        use_tqdm=True,
    )

    print(output_list[0])
    print(output_list[10])
    print(output_list[100])


if __name__ == "__main__":
    main()
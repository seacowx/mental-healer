import os
import argparse
import json, yaml
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def evaluate():
    ...


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate sentiment reward model")
    parser.add_argument(
        '--model',
        type=str,
        default='llama8',
        help='Model name to be used for evaluation'
    )
    return parser.parse_args()


def main():

    args = parse_args()

    model_path_dict = yaml.safe_load(open('../reward-finetuning/config/model_path.yaml'))
    model_path = model_path_dict[args.model]['path']

    # initialize the llm
    # llm = LLM(model=model_path, enable_lora=True)

    # TODO: Finish implementing the evaluation loop and location the best checkpoint
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=32,
    )

    test_data = json.load(open('../reward-finetuning/data/reward-sentiment_test.json'))

    input_msg_list = [
        {'role': 'user', 'content': ele['instruction'].strip()}
        for ele in test_data
    ]
    label_list = [
        ele['output'].split('<emotion>')[1].split('</emotion>')[0].strip()
        for ele in test_data
    ]

    adapter_dir = f'/scratch/prj/charnu/ft_weights/mental-healer/reward-sentiment/{args.model}/'
    lora_path_dir = os.walk(adapter_dir)

    print(lora_path_dir)
    raise SystemExit()

    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=LoRARequest("sql_adapter", 1, sql_lora_path)
    )


if __name__ == '__main__':
    main()

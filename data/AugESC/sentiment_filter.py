"""
Remove events that describe a positive sentiment as such events do not require cognitive reframing

Original Size: 65,077
After Filtering: 52,734
"""

import json, yaml
import pandas as pd
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput
from vllm.lora.request import LoRARequest


def parse_output(output: RequestOutput) -> str:

    out_str = ""
    try:
        out_str = output.outputs[0].text \
            .split('<sentiment>')[1] \
            .split('</sentiment>')[0].strip().lower() \
            .replace('"', '') \
            .replace("'", '') 
    except:
        pass

    return out_str


def classify_sentiment(data: dict) -> list:

    emotion_to_sentiment = yaml.safe_load(
        open('../../src/configs/emotion_to_sentiment.yaml', 'r'),
    )

    model_path_dict = yaml.safe_load(
        open('../../src/configs/llm_configs.yaml', 'r'),
    )
    model_path = model_path_dict['qwen7']['path']

    vllm = LLM(
        model=model_path, 
        max_model_len=2048,
        enable_lora=True,
        max_lora_rank=64,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=128,
        # stop=["</emotion>"],
    )

    adapter_dir = (
        '/scratch/prj/charnu/ft_weights/mental-healer/' 
        'reward-sentiment/qwen7/checkpoint-220'
    )

    sentiment_prompt = yaml.load(
        open('../../src/prompts/sentiment.yaml', 'r'),
        Loader=yaml.FullLoader,
    )['input']

    input_list = []
    for cur_situation in data.values():

        cur_situation_segments = [
            ele for ele in cur_situation.split('.') if ele.strip()
        ]

        if len(cur_situation_segments) > 1:
            cur_situation = '.'.join(cur_situation_segments[:-1]) + '.'
            cur_thought = cur_situation_segments[-1].strip() + '.'
        else:
            cur_situation = cur_situation_segments[0].strip() + '.'
            cur_thought = cur_situation

        input_list.append({'situation': cur_situation, 'thought': cur_thought})

    input_msg_list = [
        [{'role': 'user', 'content': sentiment_prompt.format(**ele)}]
        for ele in input_list
    ]

    outputs = vllm.chat(
        messages=input_msg_list,
        sampling_params=sampling_params,
        lora_request=LoRARequest(f"sentiment", 1, adapter_dir),
        use_tqdm=True,
    )

    sentiment_list = [parse_output(output) for output in outputs]
    sentiment_list = [
        emotion_to_sentiment.get(ele, 'none') for ele in sentiment_list
    ]
    
    return sentiment_list


def main():

    data = json.load(open('./augesc.json', 'r'))
    data_id_list = list(data.keys())
    situation_list = list(data.values())

    sentiment_list = classify_sentiment(data=data)
    negative_sentiment_id_list = [
        data_id_list[i] for i, ele in enumerate(sentiment_list) if ele == 'negative'
    ]

    data = pd.DataFrame({'situation': situation_list})
    data['sentiment'] = sentiment_list
    data.to_csv('./augesc_with_sentiment.csv', index=False)

    negative_data = data[data['sentiment'] == 'negative'
    ]
    # remove the sentiment column
    negative_data = negative_data.drop(columns=['sentiment'])
    # convert to dict
    negative_data = negative_data.to_dict(orient='records')

    out_dict = {}
    for id, entry in zip(negative_sentiment_id_list, negative_data):
        out_dict[id] = entry['situation']

    print(f"Original Size: {len(data)}")
    print(f"After Filtering: {len(negative_data)}")

    with open('./augesc_filtered.json', 'w') as f:
        json.dump(out_dict, f, indent=4)


if __name__ == "__main__":
    main()
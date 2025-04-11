import os
import time
import argparse
import json, yaml
import pandas as pd
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score


def evaluate(
    predicted: list,
    ground_truth: list,
) -> tuple:

    le = LabelEncoder()
    ground_truth_encoded = le.fit_transform(ground_truth)
    predicted_encoded = le.transform(predicted)

    accuracy = accuracy_score(ground_truth_encoded, predicted_encoded) 
    f1 = f1_score(ground_truth_encoded, predicted_encoded, average='weighted')

    return accuracy, f1


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
    llm = LLM(
        model=model_path, 
        max_model_len=2048,
        enable_lora=True,
        max_lora_rank=64,
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=128,
        # stop=["</emotion>"],
    )

    test_data = json.load(open('../reward-finetuning/data/reward-sentiment_test.json'))

    input_msg_list = [
        [{'role': 'user', 'content': ele['instruction'].strip()}]
        for ele in test_data
    ]
    label_list = [
        ele['output'].split('<emotion>')[1].split('</emotion>')[0].strip().lower()
        for ele in test_data
    ]

    adapter_dir = f'/scratch/prj/charnu/ft_weights/mental-healer/reward-sentiment/{args.model}/'
    lora_checkpoint_dir_list = [d for d in os.listdir(adapter_dir) if os.path.isdir(os.path.join(adapter_dir, d))]
    lora_checkpoint_dir_list.sort(key=lambda x: int(x.split('-')[1]))

    # load mapping from fine-grained emotion label to ternary sentiment label
    label_mapping = yaml.load(open('../reward-finetuning/data/emotion_to_sentiment.yaml'), Loader=yaml.FullLoader)

    print(label_mapping)
    raise SystemExit()
    
    eval_result_dict = {
        'lora_idx': [],
        'accuracy': [],
        'f1': []
    }
    for lora_idx, lora_checkpoint_dir in enumerate(lora_checkpoint_dir_list, 1):

        print(f"\n\nEvaluating LoRA checkpoint {lora_idx}/{len(lora_checkpoint_dir_list)}: {lora_checkpoint_dir}\n\n")
        time.sleep(1)

        cur_lora_path = os.path.join(adapter_dir, lora_checkpoint_dir)

        outputs = llm.chat(
            messages=input_msg_list,
            sampling_params=sampling_params,
            lora_request=LoRARequest(f"sentiment-[{lora_idx}]", lora_idx, cur_lora_path),
            use_tqdm=True,
        )

        # clean up the output, noise will cause issue in the label encoder
        outputs = [
            ele.outputs[0].text.split('<emotion>')[1].split('</emotion>')[0].strip().lower() \
                .replace('"', '') \
                .replace("'", '') 
            for ele in outputs
        ]

        cur_acc, cur_f1 = evaluate(
            predicted=outputs,
            ground_truth=label_list,
        )

        eval_result_dict['lora_idx'].append(lora_idx)
        eval_result_dict['accuracy'].append(cur_acc)
        eval_result_dict['f1'].append(cur_f1)

    eval_result_df = pd.DataFrame(eval_result_dict)
    print('\n\n\n')
    print(eval_result_df)


if __name__ == '__main__':
    main()

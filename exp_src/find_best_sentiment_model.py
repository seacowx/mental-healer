import os
import time
import argparse
import json, yaml
import pandas as pd

import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score


def evaluate(
    eval_idx: int, 
    predicted: list,
    ground_truth: list,
    sentiment_label_mapping: dict,
) -> tuple:

    # filter out predictions that are not in the label space
    label_space = list(set(ground_truth))
    print(label_space)
    raise SystemExit()
    valid_predicted = [ele for ele in predicted if ele in label_space]
    valid_ground_truth = [
        ground_truth[i] for i in range(len(predicted)) if predicted[i] in label_space
    ]

    # check if the number of valid predictions is less than 10% of the total
    if len(valid_predicted) < 0.8 * len(predicted):
        print(f"Warning: Less than 80% of the predictions are valid in round {eval_idx}.")

    le = LabelEncoder()
    valid_ground_truth_encoded = le.fit_transform(valid_ground_truth)
    valid_predicted_encoded = le.transform(valid_predicted)
    accuracy = accuracy_score(valid_ground_truth_encoded, valid_predicted_encoded) 
    f1 = f1_score(valid_ground_truth_encoded, valid_predicted_encoded, average='weighted')

    # evaluate sentiment (coarse-grained)
    le = LabelEncoder()
    valid_ground_truth_sentiment = [sentiment_label_mapping[ele] for ele in valid_ground_truth]
    valid_predicted_sentiment = [sentiment_label_mapping[ele] for ele in valid_predicted]
    valid_ground_truth_sentiment_encoded = le.fit_transform(valid_ground_truth_sentiment)
    valid_predicted_sentiment_encoded = le.transform(valid_predicted_sentiment)
    sentiment_accuracy = accuracy_score(valid_ground_truth_sentiment_encoded, valid_predicted_sentiment_encoded) 
    sentiment_f1 = f1_score(valid_ground_truth_sentiment_encoded, valid_predicted_sentiment_encoded, average='weighted')

    return accuracy, f1, sentiment_accuracy, sentiment_f1


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
    world_size = torch.cuda.device_count()
    llm = LLM(
        model=model_path, 
        max_model_len=2048,
        enable_lora=True,
        max_lora_rank=64,
        tensor_parallel_size=world_size,
        gpu_memory_utilization=0.8,
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
    sentiment_label_mapping = yaml.load(open('../reward-finetuning/data/emotion_to_sentiment.yaml'), Loader=yaml.FullLoader)
    
    eval_result_dict = {
        'lora_idx': [],
        'accuracy': [],
        'f1': []
    }
    sentiment_eval_result_dict = {
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
        parsed_outputs = []
        parsed_ground_truth = []
        for cur_output, cur_ground_truth in zip(outputs, label_list):
            try:
                cur_output = cur_output.outputs[0].text \
                    .split('<emotion>')[1] \
                    .split('</emotion>')[0].strip().lower() \
                    .replace('"', '') \
                    .replace("'", '') 

                parsed_outputs.append(cur_output)
                parsed_ground_truth.append(cur_ground_truth)

            except:
                pass

        cur_acc, cur_f1, cur_sentiment_acc, cur_sentiment_f1 = evaluate(
            eval_idx=lora_idx,
            predicted=parsed_outputs,
            ground_truth=parsed_ground_truth,
            sentiment_label_mapping=sentiment_label_mapping,
        )

        eval_result_dict['lora_idx'].append(lora_idx)
        eval_result_dict['accuracy'].append(cur_acc)
        eval_result_dict['f1'].append(cur_f1)
        sentiment_eval_result_dict['lora_idx'].append(lora_idx)
        sentiment_eval_result_dict['accuracy'].append(cur_sentiment_acc)
        sentiment_eval_result_dict['f1'].append(cur_sentiment_f1)

    eval_result_df = pd.DataFrame(eval_result_dict)
    sentiment_eval_result_df = pd.DataFrame(sentiment_eval_result_dict)
    print('\n\n\n')
    print(eval_result_df)
    print('\n')
    print(sentiment_eval_result_df)


if __name__ == '__main__':
    main()

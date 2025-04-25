"""
Prepare the COKE dataset for finetuning with LlamaFactory
"""

import os
import argparse
import json, yaml
import pandas as pd


def clean_text(data_list: list):

    cleaned_data_list = [
        {
            'situation': ele['situation'].replace('.', '').strip().capitalize() + '.',
            'thought': ele['thought'].replace(' .', '').strip().capitalize() + '.',
            'emotion': ele['emotion'].replace(' .', '').strip(),
        }
        for ele in data_list
    ]

    return cleaned_data_list


def assemble_data(
    data_list: list,
    emotion_to_sentiment: dict,
) -> list:
    
    instruction_template = lambda situation, thought: \
        f"<situation>\n{situation}\n</situation>\n\n<thought>\n{thought}\n</thought>"
    output_template = lambda sentiment: f"<sentiment>{sentiment}</sentiment>"

    out_data_list = []
    for entry in data_list:
        instruction = instruction_template(entry['situation'], entry['thought'])

        cur_emotion = entry.get('emotion', None)
        if cur_emotion:
            cur_emotion = entry['emotion'].replace('.', '') \
                .replace('"', '') \
                .replace("'", '').strip().lower()
            cur_sentiment = emotion_to_sentiment.get(cur_emotion, None)

            if cur_sentiment:
                output = output_template(cur_sentiment)
                out_data_list.append({
                    'instruction': instruction,
                    'input': '',
                    'output': output,
                })

    return out_data_list


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare COKE dataset for finetuning")
    parser.add_argument(
        '--model', 
        type=str,
        default='llama8',
        help='Model name to be used for finetuning'
    )
    parser.add_argument(
        '--config_path',
        type=str,
        default='../sentiment_config/sentiment_train_config1.yaml',
        help='Path to the configuration file for finetuning'
    )
    return parser.parse_args()


def prepare_for_ft():
    
    args = parse_args()
    emotion_to_sentiment = yaml.load(
        open('../../../src/configs/emotion_to_sentiment.yaml', 'r'),
        Loader=yaml.FullLoader,
    )

    # ------------------------ prepare finetuning datasets ------------------------

    coke_data = pd.read_csv('../../../data/reward-sentiment/coke/train_emotion.csv')
    coke_test_data = pd.read_csv('../../../data/reward-sentiment/coke/valid_emotion.csv')
    grouped = coke_data.groupby('situation')
    train_data = grouped.sample(frac=0.9, random_state=96)
    val_data = coke_data.drop(train_data.index)

    train_data_list = train_data.to_dict(orient='records')
    val_data_list = val_data.to_dict(orient='records')
    test_data_list = coke_test_data.to_dict(orient='records')

    train_data_list = clean_text(train_data_list)
    val_data_list = clean_text(val_data_list)
    test_data_list = clean_text(test_data_list)

    out_train_data = assemble_data(
        data_list=train_data_list,
        emotion_to_sentiment=emotion_to_sentiment,
    )
    out_val_data = assemble_data(
        data_list=val_data_list,
        emotion_to_sentiment=emotion_to_sentiment,
    )
    out_test_data = assemble_data(
        data_list=test_data_list,
        emotion_to_sentiment=emotion_to_sentiment,
    )
    
    print(f"train data: {len(out_train_data)}")
    print(f"val data: {len(out_val_data)}")
    print(f"test data: {len(out_test_data)}")

    # check if dataset_into.info needs to be updated
    updated = False

    train_path = '../sentiment_data/reward-sentiment_train.json'
    val_path = '../sentiment_data/reward-sentiment_val.json'
    test_path = '../sentiment_data/reward-sentiment_test.json'
    for path, data in zip([train_path, val_path, test_path], [out_train_data, out_val_data, out_test_data]):
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
        updated = True

    # save a copy to llamafactory data folder
    ft_root_path = '/scratch/prj/charnu/LLaMA-Factory/data/mental-healer_data'
    train_path = os.path.join(ft_root_path, 'sentiment-train.json')
    val_path = os.path.join(ft_root_path, 'sentiment-val.json')
    test_path = os.path.join(ft_root_path, 'sentiment-test.json')
    for path, data in zip([train_path, val_path, test_path], [out_train_data, out_val_data, out_test_data]):
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
        updated = True

    # load and modify the dataset info file in LLamaFactory
    dataset_info_path = '/scratch/prj/charnu/LLaMA-Factory/data/dataset_info.json'
    if updated:
        data_info_dict = json.load(
            open(dataset_info_path, 'r')
        )
        data_info_dict['mental-healer_reward-sentiment_train'] = {'file_name': '/'.join(train_path.split('/')[-2:])}
        data_info_dict['mental-healer_reward-sentiment_val'] = {'file_name': '/'.join(val_path.split('/')[-2:])}
        data_info_dict['mental-healer_reward-sentiment_test'] = {'file_name': '/'.join(test_path.split('/')[-2:])}
        with open(dataset_info_path, 'w') as f:
            json.dump(data_info_dict, f, indent=4)
        print(f"\n\nDataset info updated in {dataset_info_path}")

    # ------------------------ prepare finetuning script ------------------------

    model_path_dict = yaml.load(open('../../reward_config/model_path.yaml'), Loader=yaml.FullLoader)
    ft_script_template = yaml.load(open('../../reward_config/ft_template.yaml'), Loader=yaml.FullLoader)
    cur_config = yaml.load(open(args.config_path), Loader=yaml.FullLoader)

    assert args.model in model_path_dict, f"Model {args.model} not found in ../config/model_path.yaml"

    # fill the finetuning script with custom configurations
    cur_model_config = model_path_dict[args.model]
    ft_script_template['model_name_or_path'] = cur_model_config['path']
    ft_script_template['template'] = cur_model_config['template']
    for key, val in cur_config.items():
        ft_script_template[key] = val

    ft_script_template['dataset'] = 'mental-healer_reward-sentiment_train'

    output_fpath = os.path.join(
        '/scratch/prj/charnu/ft_weights/mental-healer/reward-sentiment',
        f'{args.model}'
    )
    if not os.path.exists(output_fpath):
        os.mkdir(output_fpath)

    ft_script_template['output_dir'] = output_fpath

    # save the prepared finetuning script
    ft_script_path = f'../sentiment_scripts/{args.model}.yaml'
    with open(ft_script_path, 'w') as f:
        yaml.dump(ft_script_template, f)
    print(f"\n\nFinetuning script saved to {ft_script_path}")


if __name__ == '__main__':
    prepare_for_ft()

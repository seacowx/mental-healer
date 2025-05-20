import json
import argparse
from tqdm import tqdm
from glob import glob


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_personas', type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    data_path_list = glob(f'./situations_with_initial_thought_top{args.n_personas}_batch*')
    data_path_list.sort()

    combined_data = {}
    unique_situation = []
    for data_path in tqdm(data_path_list, position=0, leave=False):
        with open(data_path, 'r') as f:
            data = json.load(f)

        for key, value in tqdm(data.items(), position=1, leave=False):
            combined_data[key] = value

            situation_id = key.split('||')[0].strip()
            if situation_id not in unique_situation:
                unique_situation.append(situation_id)

    print(f'Total number of situations: {len(combined_data)}')
    print(f'Total number of unique situations: {len(unique_situation)}')

    with open(f'./situations_with_initial_thought_top{args.n_personas}.json', 'w') as f:
        json.dump(combined_data, f, indent=4)


if __name__ == '__main__':
    main()



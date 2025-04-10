import json
import uuid
import numpy as np
import pandas as pd


def generate_unique_id():
    """Generate a 10-digit unique ID based on UUID."""
    # Generate UUID and take first 10 digits of its hex representation
    return str(uuid.uuid4().int)[:10]


def organize_coke(data) -> tuple:
    json_list = json.loads(json.dumps(list(data.T.to_dict().values())))
    
    visited_texts = set()
    label_dict = {}
    out_json = {}
    for entry in json_list:

        cur_context = entry['situation'].rsplit('.')[0]
        cur_thought = entry['thought'].strip()

        # remove leading "that" from cur_thought
        if cur_thought.startswith('that '):
            cur_thought = cur_thought[5:].strip()

        # remove the extra space between the last word and the period
        if cur_thought.endswith(' .'):
            cur_thought = cur_thought[:-2] + '.'

        cur_context = f"{cur_context.strip()}, and I'm thinking that {cur_thought.lower().strip()}"
        cur_emotion = entry['emotion']
        cur_emotion = cur_emotion.strip().replace('"', '').lower()

        # make sure the context is unique
        if cur_context in visited_texts:
            continue

        # make sure the label is valid
        if not cur_emotion or cur_emotion == 'emotion':
            continue

        visited_texts.add(cur_context)


        if cur_emotion not in label_dict:
            label_dict[cur_emotion] = 0

        label_dict[cur_emotion] += 1

        # produce unique id for current data entry
        cur_unique_id = generate_unique_id()
        out_json[cur_unique_id] = {
            'context': cur_context,
            'emotion_label': cur_emotion,
        }

    assert len(visited_texts) == len(out_json), "Number of unique contexts do not match!"

    return out_json, label_dict


if __name__ == '__main__':
    train_data = pd.read_csv('./train_emotion.csv')

    # split train data into train and validation with 90/10 ratio
    train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
    train_data, val_data = train_data[:int(len(train_data) * 0.9)], train_data[int(len(train_data) * 0.9):]

    coke_train_json, train_label_dict = organize_coke(train_data)
    print(f"Number of training instances: {len(coke_train_json)}")

    coke_val_json, val_label_dict = organize_coke(val_data)
    print(f"Number of validation instances: {len(coke_val_json)}")

    test_data = pd.read_csv('./valid_emotion.csv')
    coke_test_json, test_label_dict = organize_coke(test_data)

    print(f"Number of testing instances: {len(coke_test_json)}")

    assert set(train_label_dict.keys()) == set(test_label_dict.keys()) == set(val_label_dict.keys()), \
        "Train, test, validation label sets do not match!"

    with open('./coke_train.json', 'w') as f:
        json.dump(coke_train_json, f, indent=4)

    with open('./coke_val.json', 'w') as f:
        json.dump(coke_val_json, f, indent=4)

    with open('./coke_test.json', 'w') as f:
        json.dump(coke_test_json, f, indent=4)

    with open('./coke_train_label_space.json', 'w') as f:
        json.dump(train_label_dict, f, indent=4)

    with open('./coke_val_label_space.json', 'w') as f:
        json.dump(val_label_dict, f, indent=4)

    with open('./coke_test_label_space.json', 'w') as f:
        json.dump(test_label_dict, f, indent=4)

    combined_label_dict = {}
    for key, val in train_label_dict.items():
        combined_label_dict[key] = val
    for key, val in test_label_dict.items():
        combined_label_dict[key] += val

    with open('./coke_label_space.json', 'w') as f:
        json.dump(combined_label_dict, f, indent=4)


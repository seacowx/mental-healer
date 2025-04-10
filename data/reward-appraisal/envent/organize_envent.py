import uuid
import time
import json
import random
import pandas as pd


def count_labels(data_json):
    label_counts = {}
    for item in data_json.values():
        label = item['emotion_label']
        label_counts[label] = label_counts.get(label, 0) + 1
    return label_counts


def generate_unique_id():
    """Generate a 10-digit unique ID based on UUID."""
    # Generate UUID and take first 10 digits of its hex representation
    return str(uuid.uuid4().int)[:10]


data = pd.read_csv('./crowd-enVent_generation.tsv', sep='\t')
col_names = data.columns
appraisal_dims = col_names[23:44]

json_list = json.loads(json.dumps(list(data.T.to_dict().values())))

visited_texts = set()
label_dict = {}
out_json = {}
for entry in json_list:

    cur_context = entry['generated_text']

    if cur_context in visited_texts:
        continue

    # add current context to visited context, avoid duplicates
    visited_texts.add(cur_context)
    
    cur_appraisal_dims = {}
    for app_dim in appraisal_dims:
        cur_appraisal_dims[app_dim] = entry[app_dim]

    cur_emotion_label = entry['emotion']

    if cur_emotion_label not in label_dict:
        label_dict[cur_emotion_label] = 0

    label_dict[cur_emotion_label] += 1

    # produce unique id for current data entry
    cur_unique_id = generate_unique_id()
    out_json[cur_unique_id] = {
        'context': cur_context,
        'appraisal_dims': cur_appraisal_dims,
        'emotion_label': cur_emotion_label,
    }

# NOTE: check if the number of unique contexts is equal to the number of unique contexts in the output json_list
assert len(out_json) == len(visited_texts), "Number of unique contexts do not match!"

print(f'Number of unique contexts: {len(out_json)}')

with open('./envent_organized.json', 'w') as f:
    json.dump(out_json, f, indent=4)

print(f"Number of unique emotion labels: {len(label_dict)}")
with open('./envent_label_space.json', 'w') as f:
    json.dump(label_dict, f, indent=4)

# NOTE: train-test split
test_fraction = 0.1
val_fraction = 0.1

# split data based in ids
random.seed(96)
all_keys = list(out_json.keys())
test_keys = random.sample(all_keys, int(test_fraction * len(all_keys)))
remaining_keys = list(set(all_keys) - set(test_keys))
val_keys = random.sample(remaining_keys, int(val_fraction * len(all_keys)))
remaining_keys = list(set(remaining_keys) - set(val_keys))

test_json = {k: out_json[k] for k in test_keys}
val_json = {k: out_json[k] for k in val_keys}
train_json = {k: out_json[k] for k in remaining_keys}

train_labels = set([train_json[k]['emotion_label'] for k in train_json])
val_labels = set([val_json[k]['emotion_label'] for k in val_json])
test_labels = set([test_json[k]['emotion_label'] for k in test_json])

assert train_labels == val_labels == test_labels, "Labels in train, validation, and test sets are not the same!"

# Count labels in each split
train_label_counts = count_labels(train_json)
val_label_counts = count_labels(val_json)
test_label_counts = count_labels(test_json)

# Make sure that the keys in these subsets are not overlapping
assert len(set(test_json.keys()) & set(val_json.keys())) == 0, "Test and validation keys are overlapping!"
assert len(set(test_json.keys()) & set(train_json.keys())) == 0, "Test and train keys are overlapping!"
assert len(set(val_json.keys()) & set(train_json.keys())) == 0, "Validation and train keys are overlapping!"

print(f'Number of training samples: {len(train_json)}')
with open('./envent_train.json', 'w') as f:
    json.dump(train_json, f, indent=4)

print(f'Number of validation samples: {len(val_json)}')
with open('./envent_val.json', 'w') as f:
    json.dump(val_json, f, indent=4)

print(f'Number of test samples: {len(test_json)}')
with open('./envent_test.json', 'w') as f:
    json.dump(test_json, f, indent=4)

with open('./envent_train_label_space.json', 'w') as f:
    json.dump(train_label_counts, f, indent=4)

with open('./envent_val_label_space.json', 'w') as f:
    json.dump(val_label_counts, f, indent=4)

with open('./envent_test_label_space.json', 'w') as f:
    json.dump(test_label_counts, f, indent=4)


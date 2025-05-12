import json, yaml

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score


def evaluate(
    eval_idx: int, 
    predicted: list,
    ground_truth: list,
) -> tuple:

    sentiment_label_mapping = yaml.load(
        open('../src/configs/emotion_to_sentiment.yaml'),
        Loader=yaml.FullLoader
    )

    # filter out predictions that are not in the label space
    label_space = list(set(ground_truth))
    valid_predicted = [ele for ele in predicted if ele in label_space]
    valid_ground_truth = [
        gt_ele for (gt_ele, pred_ele) in zip(ground_truth, predicted) if pred_ele in label_space
    ]

    # check if the number of valid predictions is less than 10% of the total
    if len(valid_predicted) < 0.8 * len(predicted):
        print(f"Warning: Less than 80% of the predictions are valid in round {eval_idx}.")
        return -1, -1, -1, -1

    le = LabelEncoder()
    valid_ground_truth_encoded = le.fit_transform(valid_ground_truth)
    valid_predicted_encoded = le.transform(valid_predicted)

    accuracy = accuracy_score(
        valid_ground_truth_encoded, 
        valid_predicted_encoded
    ) 
    f1 = f1_score(
        valid_ground_truth_encoded, 
        valid_predicted_encoded, 
        average='weighted',
    )

    # evaluate sentiment (coarse-grained)
    le = LabelEncoder()
    valid_ground_truth_sentiment = [sentiment_label_mapping[ele] for ele in valid_ground_truth]
    valid_predicted_sentiment = [sentiment_label_mapping[ele] for ele in valid_predicted]
    valid_ground_truth_sentiment_encoded = le.fit_transform(valid_ground_truth_sentiment)
    valid_predicted_sentiment_encoded = le.transform(valid_predicted_sentiment)

    sentiment_accuracy = accuracy_score(
        valid_ground_truth_sentiment_encoded, 
        valid_predicted_sentiment_encoded
    ) 
    sentiment_f1 = f1_score(
        valid_ground_truth_sentiment_encoded, 
        valid_predicted_sentiment_encoded, 
        average='weighted'
    )

    return accuracy, f1, sentiment_accuracy, sentiment_f1



def test_sentiment(sentiment_reward_model):

    test_data = json.load(open('../reward_ft/reward-sentiment/sentiment_data/reward-sentiment_test.json'))
    sentiment_label_mapping = yaml.safe_load(open('./configs/emotion_to_sentiment.yaml'))

    input_msg_list = [
        [{'role': 'user', 'content': ele['instruction'].strip()}]
        for ele in test_data
    ]
    label_list = [
        ele['output'].split('<sentiment>')[1].split('</sentiment>')[0].strip().lower()
        for ele in test_data
    ]

    outputs = sentiment_reward_model.get_sentiment(
        input_msg_list
    )

    # clean up the output, noise will cause issue in the label encoder
    parsed_outputs = []
    parsed_ground_truth = []
    for cur_output, cur_ground_truth in zip(outputs, label_list):
        cur_output = cur_output.outputs[0].text \
            .split('<sentiment>')[1] \
            .split('</sentiment>')[0].strip().lower() \
            .replace('"', '') \
            .replace("'", '') 

        parsed_outputs.append(cur_output)
        parsed_ground_truth.append(cur_ground_truth)

    cur_acc, cur_f1, cur_sentiment_acc, cur_sentiment_f1 = evaluate(
        eval_idx=1,
        predicted=parsed_outputs,
        ground_truth=parsed_ground_truth,
    )

    print(f"Accuracy: {cur_acc}, F1: {cur_f1}, Sentiment Accuracy: {cur_sentiment_acc}, Sentiment F1: {cur_sentiment_f1}")

    raise SystemExit
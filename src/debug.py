import json, yaml

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score


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
        input_msg_list[:10]
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

    print(parsed_outputs)
    print(parsed_ground_truth)
    raise SystemExit

    # evaluate sentiment (coarse-grained)
    le = LabelEncoder()
    valid_ground_truth_sentiment = [sentiment_label_mapping[ele] for ele in parsed_ground_truth]
    valid_ground_truth_sentiment_encoded = le.fit_transform(valid_ground_truth_sentiment)
    valid_predicted_sentiment_encoded = le.transform(parsed_outputs)

    # print(f"Accuracy: {cur_acc}, F1: {cur_f1}, Sentiment Accuracy: {cur_sentiment_acc}, Sentiment F1: {cur_sentiment_f1}")

    raise SystemExit
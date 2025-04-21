import yaml
from rewards.sentiment import SentimentReward
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score


# WARNING: remove after testing
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


class TherapistTrainer:


    def __init__(self, data: list) -> None:
        self.data = data
        self.sentiment_prompt = yaml.load(
            open('./prompts/sentiment.yaml', 'r'),
            Loader=yaml.FullLoader,
        )['input']
        self.sentiment_reward = SentimentReward()


    def __compute_sentiment_reward(
        self, 
        input_list: list,
        label_list: list,
    ) -> list:

        # # make prompt
        # input_msg_list = [
        #     [{'role': 'user', 'content': self.sentiment_prompt.format(**ele)}]
        #     for ele in input_list
        # ]

        sentiment_list = self.sentiment_reward.get_sentiment(
            input_msg_list=input_list,
        )

        evaluate(
            eval_idx=0,
            predicted=sentiment_list,
            ground_truth=label_list,
        )

        raise SystemExit()

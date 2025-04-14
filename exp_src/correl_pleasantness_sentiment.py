import json, yaml
from sklearn.metrics import accuracy_score, f1_score


def sanity_check(
    pleasantness_rating: float, 
    unpleasantness_rating: float,
):
    if pleasantness_rating > 3 and unpleasantness_rating > 3:
        return False
    elif pleasantness_rating < 3 and unpleasantness_rating < 3:
        return False
    return True


def main():

    envent_data = json.load(
        open('../data/reward-appraisal/envent/envent_organized.json', 'r')
    )
    emotion_to_sentiment_dict = yaml.load(
        open('../reward/reward-sentiment/sentiment_data/emotion_to_sentiment.yaml', 'r'),
        Loader=yaml.FullLoader,
    )

    sentiment_labels = []
    pleasantness_labels = []
    for key, entry in envent_data.items():

        cur_pleasantness_rating = entry['appraisal_dims']['pleasantness']
        cur_unpleasantness_rating = entry['appraisal_dims']['unpleasantness']

        cur_emotion_label = entry['emotion_label']
        cur_sentiment_label = emotion_to_sentiment_dict[cur_emotion_label]

        valid_entry = sanity_check(
            pleasantness_rating=cur_pleasantness_rating,
            unpleasantness_rating=cur_unpleasantness_rating,
        )

        if not valid_entry:
            continue

        cur_pleasantness_label = 1 if cur_pleasantness_rating > cur_unpleasantness_rating else 0
        cur_sentiment_label = 1 if cur_sentiment_label == 'positive' else 0

        sentiment_labels.append(cur_sentiment_label)
        pleasantness_labels.append(cur_pleasantness_label)

    acc = accuracy_score(
        y_true=sentiment_labels,
        y_pred=pleasantness_labels,
    )
    f1 = f1_score(
        y_true=sentiment_labels,
        y_pred=pleasantness_labels,
    )

    print(f'Accuracy: {acc}')
    print(f'F1: {f1}')


if __name__ == '__main__':
    main()

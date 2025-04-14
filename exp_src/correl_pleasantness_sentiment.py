import json, yaml
from sklearn.metrics import accuracy_score, f1_score


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
        print(entry)
        raise SystemExit()


if __name__ == '__main__':
    main()

import re
import yaml
import pandas as pd


def clean_emotion(emotion):

    # remove non-alphabetical characters and convert to lowercase
    return ''.join(filter(str.isalpha, emotion)).lower()


def main():

    emotion_to_sentiment = yaml.load(
        open('../../src/configs/emotion_to_sentiment.yaml', 'r'),
        Loader=yaml.FullLoader,
    )

    coke_train = pd.read_csv("../reward-sentiment/coke/train_emotion.csv")
    coek_val = pd.read_csv("../reward-sentiment/coke/valid_emotion.csv")

    coke_data = pd.concat([coke_train, coek_val], axis=0)
    coke_data['emotion'] = coke_data['emotion'].apply(clean_emotion)
    coke_data['emotion'] = coke_data['emotion'].map(emotion_to_sentiment)

    coke_negative_thought_data = coke_data[coke_data['emotion'] == 'negative']
    coke_negative_thought_data['situation'] = coke_negative_thought_data['situation'].apply(
        lambda x: re.sub(r'\s+\.', '.', x).strip()
    )
    coke_negative_thought_data['thought'] = coke_negative_thought_data['thought'].apply(
        lambda x: re.sub(r'\s+\.', '.', x).strip()
    )

    data_part1 = coke_negative_thought_data[['situation', 'thought']]
    data_part1['thinking_traps_addressed'] = 'Unlabeled'

    # ----------------------------------------------------------------------------------------

    data_part2 = pd.read_csv('../../data/acl23-reframing/reframing_dataset.csv')
    data_part2 = data_part2[['situation', 'thought', 'thinking_traps_addressed']]

    data_part2['situation'] = data_part2['situation'].apply(
        lambda x: re.sub(r'\s+\.', '.', x).strip()
    )
    data_part2['thought'] = data_part2['thought'].apply(
        lambda x: re.sub(r'\s+\.', '.', x).strip()
    )

    # ----------------------------------------------------------------------------------------

    out_data = pd.concat([data_part1, data_part2], axis=0)
    out_data.to_csv('./situations.csv', index=False)

if __name__ == "__main__":
    main()

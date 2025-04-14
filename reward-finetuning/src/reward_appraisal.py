import json, yaml
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def clustering():

    envent_data = json.load(open('../../data/reward-appraisal/envent/envent_organized.json'))
    emotion_to_sentiment = yaml.load(
        open('../data/emotion_to_sentiment.yaml'), 
        Loader=yaml.FullLoader
    )

    # record appraisal profiles
    appraisal_mtx = np.zeros((len(envent_data), 21))
    emotion_labels = []
    for idx, val in enumerate(envent_data.values()):
        cur_appraisal_profile = list(val['appraisal_dims'].values())
        appraisal_mtx[idx] = cur_appraisal_profile
        emotion_labels.append(val['emotion_label'])

    sentiment_labels = [
        emotion_to_sentiment[label] for label in emotion_labels
    ]

    kmeans = KMeans(
        n_clusters=2, 
        random_state=0, 
    )

    kmeans.fit(appraisal_mtx)
    labels = kmeans.labels_

    organized_results = {
        'cluster': labels,
        'sentiment': sentiment_labels,
    }

    out_df = pd.DataFrame(organized_results)
    out_df.to_csv('../../exp_data/cluster_result.csv', index=False)


if __name__ == '__main__':
    clustering()

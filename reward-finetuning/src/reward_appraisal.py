import json, yaml
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score


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

    # organized_results = {
    #     'cluster': labels,
    #     'sentiment': sentiment_labels,
    # }
    #
    # out_df = pd.DataFrame(organized_results)
    # out_df.to_csv('../../exp_data/cluster_result.csv', index=False)
    #
    pred_labels = deepcopy(labels)
    sentiment_labels = [
        1 if label == 'negative' else 0 for label in sentiment_labels
    ]

    print(f"accuracy: {accuracy_score(sentiment_labels, pred_labels)}")
    print(f"f1: {f1_score(sentiment_labels, pred_labels)}") 

    appraisal_embedded = TSNE(n_components=2, learning_rate='auto',
                  init='random', perplexity=3).fit_transform(appraisal_mtx)

    # print(appraisal_embedded)


if __name__ == '__main__':
    clustering()

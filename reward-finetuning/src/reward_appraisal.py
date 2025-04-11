import json
import numpy as np
from sklearn.cluster import KMeans


def clustering():

    envent_data = json.load(open('../../data/reward-appraisal/envent/envent_organized.json'))

    # record appraisal profiles
    appraisal_mtx = np.zeros((len(envent_data), 21))
    emotion_labels = []
    for idx, val in enumerate(envent_data.values()):
        cur_appraisal_profile = list(val['appraisal_dims'].values())
        appraisal_mtx[idx] = cur_appraisal_profile
        emotion_labels.append(val['emotion_label'])

    kmeans = KMeans(n_clusters=2, random_state=0)
    print(appraisal_mtx)


if __name__ == '__main__':
    clustering()

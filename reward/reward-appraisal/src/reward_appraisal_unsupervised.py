import os
import argparse
import json, yaml
import numpy as np
import pandas as pd
from copy import deepcopy

import plotly.express as px
import matplotlib.pyplot as plt

import torch
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score

from reward_appraisal_predict import AppraisalPredictor


def down_project_data(appraisal_mtx: np.ndarray):

    appraisal_mtx_embedded = TSNE(
        n_components=2, 
        learning_rate='auto',
        init='random', 
        perplexity=20,
        random_state=12,
    ).fit_transform(appraisal_mtx)

    return appraisal_mtx_embedded


def clustering(appraisal_mtx: np.ndarray):
    kmeans = KMeans(
        n_clusters=2, 
        random_state=12, 
    )

    kmeans.fit(appraisal_mtx)
    return kmeans.labels_


def evaluate(labels: list, sentiment_labels: list) -> tuple:
    result_save_fpath = '../../../exp_data/appraisal_clustering_result.json'
    if not os.path.exists(result_save_fpath):
        organized_results = {
            'cluster': labels,
            'sentiment': sentiment_labels,
        }

        out_df = pd.DataFrame(organized_results)
        out_df.to_csv(result_save_fpath, index=False)

    pred_labels = deepcopy(labels)
    numerical_sentiment_labels = [
        1 if label == 'negative' else 0 for label in sentiment_labels
    ]

    accuracy = accuracy_score(numerical_sentiment_labels, pred_labels)
    f1 = f1_score(numerical_sentiment_labels, pred_labels)

    return accuracy, f1


def visualize_clustered_data(vis_data: pd.DataFrame):
    # Create scatter plot with Plotly
    fig = px.scatter(
        vis_data,
        x='x',
        y='y',
        color='sentiment',
        title="Appraisal Embedded Scatter Plot",
        labels={'x': 'Dimension 1', 'y': 'Dimension 2'},
    )
    fig.update_traces(marker=dict(size=8, opacity=0.75, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(legend_title_text='Sentiment')
    fig.write_image(f"../../../exp_data/appraisal_clustering.png")
    fig.write_html(f"../../../exp_data/appraisal_clustering.html")
    fig.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare COKE dataset for finetuning")
    parser.add_argument(
        '--model', 
        type=str,
        default='llama8',
        help='Model name to be used for rating appraisals'
    )
    parser.add_argument(
        '--predict_appraisal',
        action='store_true',
        help='Flag to indicate if appraisal prediction is needed, otherwise use human annotating'
    )
    return parser.parse_args()


def main():

    args = parse_args()

    if args.predict_appraisal:
        assert torch.cuda.is_available(), "CUDA is not available. Appraisal prediction requires GPU."

    envent_data = json.load(open('../../../data/reward-appraisal/envent/envent_organized.json'))
    emotion_to_sentiment = yaml.load(
        open('../../reward-sentiment/sentiment_data/emotion_to_sentiment.yaml'), 
        Loader=yaml.FullLoader
    )

    appraisal_mtx = np.zeros((len(envent_data), 21))
    emotion_labels = []
    if args.predict_appraisal:
        prompt_template = json.load(
            open('../prompts/predict_appraisal.yaml', 'r')
        )
        appraisal_predictor = AppraisalPredictor(
            prompt_template=prompt_template,
        )
        predicted_apprarisal_profiles = appraisal_predictor.predict()
    else:
        # retrieve human annotation from EnVent dataset
        for idx, val in enumerate(envent_data.values()):
            cur_appraisal_profile = list(val['appraisal_dims'].values())
            appraisal_mtx[idx] = cur_appraisal_profile
            emotion_labels.append(val['emotion_label'])

    sentiment_labels = [
        emotion_to_sentiment[label] for label in emotion_labels
    ]

    appraisal_mtx_embedded = down_project_data(
        appraisal_mtx=appraisal_mtx,
    )

    # make binary sentiment label for visualization
    binary_sentiment_labels = [
        'negative' if label == 'negative' else 'positive' for label in sentiment_labels
    ]

    clustering_labels = clustering(
        appraisal_mtx=appraisal_mtx,
    )
    embedded_clustering_labels = clustering(
        appraisal_mtx=appraisal_mtx_embedded,
    )

    acc, f1 = evaluate(
        labels=clustering_labels,
        sentiment_labels=sentiment_labels,
    )
    dp_acc, dp_f1 = evaluate(
        labels=embedded_clustering_labels,
        sentiment_labels=sentiment_labels,
    )

    print("\n\nEvaluation on Full Appraisal Profile (21d)")
    print(f"accuracy: {acc}")
    print(f"f1: {f1}") 
    print("\nEvaluation on T-SNE Embedded Appraisal Profile (2d)")
    print(f"accuracy: {dp_acc}")
    print(f"f1: {dp_f1}") 

    # visualize the down-projected data
    vis_data = pd.DataFrame({ 
        'x': [point[0] for point in appraisal_mtx_embedded],
        'y': [point[1] for point in appraisal_mtx_embedded],
        'sentiment': binary_sentiment_labels,
    })
    visualize_clustered_data(
        vis_data=vis_data
    )


if __name__ == '__main__':
    main()

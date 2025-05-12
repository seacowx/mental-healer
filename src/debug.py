import json, yaml


async def test_sentiment(sentiment_reward_model):

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

    outputs = await sentiment_reward_model.get_sentiment(
        input_msg_list
    )
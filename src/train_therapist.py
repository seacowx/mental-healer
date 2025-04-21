import yaml
from rewards.sentiment import SentimentReward


class TherapistTrainer:


    def __init__(self, data: list) -> None:
        self.data = data
        self.sentiment_prompt = yaml.load(
            open('./prompts/sentiment.yaml', 'r'),
            Loader=yaml.FullLoader,
        )['input']
        self.sentiment_reward = SentimentReward()


    def __compute_sentiment_reward(self, input_list: list) -> list:

        # make prompt
        input_msg_list = [
            [{'role': 'user', 'content': self.sentiment_prompt.format(**ele)}]
            for ele in input_list
        ]

        sentiment_list = self.sentiment_reward.get_sentiment(
            input_msg_list=input_msg_list,
        )

        print(sentiment_list)
        raise SystemExit()

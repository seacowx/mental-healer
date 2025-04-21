import yaml

from rewards.sentiment import SentimentReward
from rewards.semantic_similarity import SemanticSimilarityReward

class TherapistReward:


    def __init__(self) -> None:
        self.sentiment_prompt = yaml.load(
            open('./prompts/sentiment.yaml', 'r'),
            Loader=yaml.FullLoader,
        )['input']
        self.sentiment_reward = SentimentReward()
        self.semantic_similarity_reward = SemanticSimilarityReward()


    def __reward_sentiment(
        self, 
        input_list: list,
    ) -> list:

        # make prompt
        input_msg_list = [
            [{'role': 'user', 'content': self.sentiment_prompt.format(**ele)}]
            for ele in input_list
        ]

        sentiment_list = self.sentiment_reward.get_sentiment(
            input_msg_list=input_msg_list,
        )

        return sentiment_list

    
    def __reward_semantic_similarity(
        self,
        utterance_list: list,
        response_list: list,
    ):
        raise NotImplementedError()


    def compute_reward(
        self, 
        situation_list: list,
        old_thought_list: list,
        new_thought_list: list,
        utterance_list: list,
    ):
        ...

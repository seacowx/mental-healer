import yaml

import torch
from rewards.sentiment import SentimentReward
from rewards.semantic_similarity import SemanticSimilarityReward

class TherapistReward:

    def __init__(self, sentiment_reward_device: torch.device) -> None:
        self.sentiment_prompt = yaml.load(
            open('./prompts/sentiment.yaml', 'r'),
            Loader=yaml.FullLoader,
        )['input']
        self.sentiment_reward = SentimentReward(
            sentiment_reward_device=sentiment_reward_device,
        )
        self.semantic_similarity_reward = SemanticSimilarityReward()

        self.SENTIMENT_COEFFICIENT = 0.7
        self.SEMANTIC_SIMILARITY_COEFFICIENT = 0.3
    

    @classmethod
    def set_coefficients(
        cls,
        sentiment_coefficient: float,
        semantic_similarity_coefficient: float,
    ):
        cls.SENTIMENT_COEFFICIENT = sentiment_coefficient
        cls.SEMANTIC_SIMILARITY_COEFFICIENT = semantic_similarity_coefficient


    @property
    def coefficients(self):
        return {
            'sentiment_coefficient': self.SENTIMENT_COEFFICIENT,
            'semantic_similarity_coefficient': self.SEMANTIC_SIMILARITY_COEFFICIENT,
        }

    
    def make_sentiment_input_msg(self, situation_list: list, thoutght_list: list) -> list:
        input_list = [
            {'situation': situation, 'thought': thought,}
            for situation, thought in zip(situation_list, thoutght_list)
        ]

        # make prompt
        input_msg_list = [
            [{'role': 'user', 'content': self.sentiment_prompt.format(**ele)}]
            for ele in input_list
        ]

        return input_msg_list   


    def reward_sentiment(
        self, 
        situation_list: list,
        thoutght_list: list,
        previous_sentiment_list: list,
    ) -> tuple[list, list]:

        input_msg_list = self.make_sentiment_input_msg(
            situation_list=situation_list,
            thoutght_list=thoutght_list,
        )

        new_sentiment_list = self.sentiment_reward.get_sentiment(
            input_msg_list=input_msg_list,
        )

        sentiment_reward_list = []
        if previous_sentiment_list:
            sentiment_reward_list = self.sentiment_reward.compute_sentiment_reward(
                new_sentiment_list=new_sentiment_list,
                previous_sentiment_list=previous_sentiment_list,
            )

        return new_sentiment_list, sentiment_reward_list

    
    def reward_semantic_similarity(
        self,
        utterance_list: list,
        response_list: list,
    ):
        return self.semantic_similarity_reward.compute_similarity(
            utterance_list=utterance_list,
            response_list=response_list,
        )


    def compute_reward(
        self, 
        situation_list: list,
        old_thought_list: list,
        new_thought_list: list,
        utterance_list: list,
        previous_sentiment_list: list,
    ):

        new_sentiment_list, sentiment_reward_list = self.reward_sentiment(
            situation_list=situation_list,
            thoutght_list=new_thought_list,
            previous_sentiment_list=previous_sentiment_list,
        )

        semantic_similarity_list = self.reward_semantic_similarity(
            utterance_list=utterance_list,
            response_list=new_thought_list,
        ) 

        # TODO: find appropriate threshold for semantic similarity rewards
        # TODO: add coefficient to balance the two rewards

        return [
            self.SENTIMENT_COEFFICIENT * sentiment_reward + 
            self.SEMANTIC_SIMILARITY_COEFFICIENT * semantic_similarity
            for sentiment_reward, semantic_similarity in zip(sentiment_reward_list, semantic_similarity_list)
        ]


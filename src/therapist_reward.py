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
        situation_list: list,
        thoutght_list: list,
        previous_sentiment_list: list,
    ) -> list:

        input_list = [
            {'situation': situation, 'thought': thought,}
            for situation, thought in zip(situation_list, thoutght_list)
        ]

        # make prompt
        input_msg_list = [
            [{'role': 'user', 'content': self.sentiment_prompt.format(**ele)}]
            for ele in input_list
        ]

        sentiment_list = self.sentiment_reward.get_sentiment(
            input_msg_list=input_msg_list,
            previous_sentiment_list=previous_sentiment_list,
        )

        return sentiment_list

    
    def __reward_semantic_similarity(
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

        sentiment_list = self.__reward_sentiment(
            situation_list=situation_list,
            thoutght_list=new_thought_list,
            previous_sentiment_list=previous_sentiment_list,
        )

        semantic_similarity_list = self.__reward_semantic_similarity(
            utterance_list=utterance_list,
            response_list=new_thought_list,
        ) 
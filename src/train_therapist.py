from rewards.sentiment import Sentiment


class TherapistTrainer:


    def __init__(self, data: list) -> None:
        self.data = data
        self.sentiment_reward = Sentiment()

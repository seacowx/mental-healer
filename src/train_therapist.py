from rewards.sentiment import SentimentReward


class TherapistTrainer:


    def __init__(self, data: list) -> None:
        self.data = data
        self.sentiment_reward = SentimentReward()

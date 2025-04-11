class MentalReward():

    def __init__(self, reward: float = 0.0):
        self.reward = reward

    def sentiment_reward(self) -> float:
        raise NotImplementedError()


    def appraisal_reward(self):
        raise NotImplementedError()


    def similarity_reward(self):
        raise NotImplementedError()
    

    def aggregate_rewards(self, rewards: list) -> float:
        """
        Aggregate the rewards from the list of rewards.
        """
        ...

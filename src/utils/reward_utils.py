from rewards.sentiment import SentimentReward
from utils.vllm_inference_utils import OpenAIAsyncInference


def initialize_sentiment_reward_model(
    client_port: int,
):

    sentiment_reward_model = SentimentReward(
        base_vllm_client=base_client,
    )

    return sentiment_reward_model


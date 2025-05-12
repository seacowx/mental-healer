from rewards.sentiment import SentimentReward
from utils.vllm_inference_utils import vLLMServer


def initialize_sentiment_reward_model(
    base_server: vLLMServer,
):

    sentiment_reward_model = SentimentReward(
        base_vllm_server=base_server,
    )

    return sentiment_reward_model


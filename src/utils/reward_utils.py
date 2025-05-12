import torch
from vllm import LLM

from rewards.sentiment import SentimentReward
from utils.vllm_inference_utils import OpenAIAsyncInference


def initialize_sentiment_reward_model(
    model_path: str,
    sentiment_reward_device: torch.device,
) -> LLM:

    sentiment_reward_model = SentimentReward(
        model_path=model_path,
        sentiment_reward_device=sentiment_reward_device,
    )

    return sentiment_reward_model.initialize_sentiment_reward_model()


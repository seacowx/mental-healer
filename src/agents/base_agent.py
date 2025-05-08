from abc import ABCMeta, abstractmethod
from src.utils.vllm_inference_utils import vLLMServer


class LMAgent(metaclass=ABCMeta):
    """
    LMAgent class for async inference with OpenAI APIError
    """

    def __init__(self, client, async_client, vllm_client) -> None:
        self.client = client
        self.async_client = async_client
        self.vllm_client = vllm_client


    @abstractmethod
    def utter(self) -> str:
        """
        Produce utterance
        """
        pass

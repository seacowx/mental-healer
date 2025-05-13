from abc import ABCMeta, abstractmethod
from utils.vllm_inference_utils import vLLMServer


class LMAgent(metaclass=ABCMeta):
    """
    LMAgent class for async inference with OpenAI APIError
    """

    def __init__(self, client, async_client, base_vllm_model) -> None:
        self.client = client
        self.async_client = async_client
        self.base_vllm_model = base_vllm_model


    @abstractmethod
    def utter(self) -> str:
        """
        Produce utterance
        """
        pass

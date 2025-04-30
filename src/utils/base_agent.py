from abc import ABCMeta, abstractmethod
from utils.llm_inference import vLLMServer


class LMAgent(metaclass=ABCMeta):
    """
    LMAgent class for async inference with OpenAI APIError
    """

    def __init__(self, client, async_client):
        self.client = client
        self.async_client = async_client


    @abstractmethod
    def utter(self) -> str:
        """
        Produce utterance
        """
        pass

from abc import ABCMeta, abstractmethod
from utils.vllm_inference_utils import vLLMServer


class LMAgent(metaclass=ABCMeta):
    """
    LMAgent class for async inference with OpenAI APIError
    """

    def __init__(
        self, 
        openai_client, 
        openai_async_client, 
        base_vllm_model
    ) -> None:

        assert (openai_client is not None) or \
            (openai_async_client is not None) or \
            (base_vllm_model is not None), \
            "at least one of openai_client, openai_async_client, or base_vllm_model must be provided"

        self.openai_client = openai_client
        self.openai_async_client = openai_async_client
        self.base_vllm_model = base_vllm_model


    @abstractmethod
    def utter(self) -> str:
        """
        Produce utterance
        """
        pass

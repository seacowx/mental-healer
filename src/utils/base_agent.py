from abc import ABCMeta, abstractmethod
from utils.llm_inference import vLLMServer


class LMAgent(metaclass=ABCMeta):
    """
    LMAgent class for async inference with OpenAI APIError
    """

    def __init__(self, **kwargs):
        self.async_client = None
        self.server = vLLMServer(**kwargs)


    def init_model(self) -> None:
        """
        Initialize the agent
        """
        self.async_client = self.server.start_vllm_server()


    def terminate_model(self) -> None:
        """
        Terminate the agent
        """
        self.server.kill_server()


    @abstractmethod
    def utter(self) -> str:
        """
        Produce utterance
        """
        pass

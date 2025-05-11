from agents.base_agent import LMAgent


class TherapistAgent(LMAgent):

    def __init__(
        self,
        vllm_client: object = None,
        openai_client: object = None,
        openai_async_client: object = None,
    ) -> None:
        super().__init__(
            client=openai_client, 
            async_client=openai_async_client,
            vllm_client=vllm_client,
        )


    def utter(self, patient_utterance: str) -> str:
        raise NotImplementedError()
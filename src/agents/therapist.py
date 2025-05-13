from agents.base_agent import LMAgent


class TherapistAgent(LMAgent):

    def __init__(
        self,
        base_vllm_model: object = None,
        openai_client: object = None,
        openai_async_client: object = None,
    ) -> None:

        super().__init__(
            openai_client=openai_client, 
            openai_async_client=openai_async_client,
            base_vllm_model=base_vllm_model,
        )


    def utter(self, patient_utterance: str) -> str:
        raise NotImplementedError()
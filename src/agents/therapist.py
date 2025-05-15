from agents.base_agent import LMAgent
from utils.persona_utils import verbalize_persona_profile
from utils.therapeutic_utils import TherapeuticSessionBuffer


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


    def utter(
        self, 
        situation_desc_list: list[str],
        patient_thought_list: list[str],
        patient_persona_profile_list: list[str],
        session_buffer: TherapeuticSessionBuffer, 
    ) -> list[dict]:

        # verbalize the persona profile
        patient_persona_profile_desc_list = [
            verbalize_persona_profile(
                persona_profile_dict=ele
            )
            for ele in patient_persona_profile_list
        ]

        utterance_list = self.base_vllm_model.inference(
            situation_desc_list=situation_desc_list,
            patient_thought_list=patient_thought_list,
            patient_persona_profile_desc_list=patient_persona_profile_desc_list,
            session_buffer=session_buffer,
            is_coping_utterance=True,
        )

        return utterance_list
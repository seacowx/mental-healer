"""
Patient Agent. Frozen during RL training.

This agent is responsible for generating the initial thought and updating the thought based on the therapist's utterance.
"""

import os
import operator
import yaml, json
import pandas as pd
from copy import deepcopy

from openai import OpenAI, AsyncOpenAI

from agents.base_agent import LMAgent
from utils.persona_utils import verbalize_persona_profile
from utils.therapeutic_utils import TherapeuticSessionBuffer
from utils.vllm_inference_utils import vLLMOffline, OpenAIAsyncInference

class PatientAgent(LMAgent):


    def __init__(
        self,
        base_vllm_model: vLLMOffline | None = None,
        openai_client: OpenAI | None = None,
        openai_async_client: AsyncOpenAI | OpenAIAsyncInference | None = None,
        patient_template_path: str = './prompts/patient.yaml',
    ) -> None:

        super().__init__(
            openai_client=openai_client, 
            openai_async_client=openai_async_client,
            base_vllm_model=base_vllm_model,
        )

        self.meta_persona_profile = []
        self.role_playing_instruction_list = []
        self.patient_template = yaml.safe_load(
            open(patient_template_path)
        )


    @property
    def current_persona_profile(self) -> str:
        """
        Get the persona profile for the agent
        """
        return json.dumps(self.meta_persona_profile)


    def set_persona(self, persona_profile_dict_list: list[dict]) -> None:
        """
        Set the persona profile for the agent
        """
        self.meta_persona_profile = persona_profile_dict_list
        self.role_playing_instruction_list = [ele['persona_hub'] for ele in self.meta_persona_profile]
        self.persona_profile_dict_list = [
            {key: val for key, val in ele.items() if key != 'persona_hub'} for ele in self.meta_persona_profile
        ]


    def utter(
        self, 
        situation_desc_list: list[str],
        patient_thought_list: list[str],
        session_buffer: TherapeuticSessionBuffer,
    ) -> str:

        assert self.meta_persona_profile, \
            (
                "Persona profile is not set. Please set it using " 
                "'set_persona(persona_profile_dict_list)' before calling the utter method."
            )

        patient_new_thought_msg = []
        for sample_idx in range(len(situation_desc_list)):

            cur_persona_profile = self.meta_persona_profile[sample_idx]
            print(cur_persona_profile)
            raise SystemExit

            cur_session_history = session_buffer.get_session_history(sample_idx=sample_idx)

            cur_persona_profile_desc = verbalize_persona_profile(
                persona_profile_dict=cur_persona_profile
            )

            print(cur_persona_profile_desc)
            raise SystemExit

        # prompt needs: persona_profile, situation, thought, therapist_utterance
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
from utils.vllm_inference_utils import vLLMOffline, OpenAIAsyncInference


class PatientAgent(LMAgent):


    def __init__(
        self,
        base_vllm_model: vLLMOffline | None = None,
        openai_client: OpenAI | None = None,
        openai_async_client: AsyncOpenAI | OpenAIAsyncInference | None = None,
        update_thought_template_path: str = './prompts/update_thought.yaml',
    ) -> None:

        super().__init__(
            openai_client=openai_client, 
            openai_async_client=openai_async_client,
            base_vllm_model=base_vllm_model,
        )

        self.meta_persona_profile = {}
        self.persona_profile = {}
        self.role_playing_instruction = ''
        self.update_thought_template = yaml.safe_load(
            open(update_thought_template_path)
        )


    @property
    def show_persona_profile(self) -> str:
        """
        Get the persona profile for the agent
        """
        return json.dumps(self.meta_persona_profile)


    def set_persona(self, persona_profile: dict) -> None:
        """
        Set the persona profile for the agent
        """
        self.meta_persona_profile = persona_profile
        self.role_play_instruction = self.meta_persona_profile['persona_hub']
        self.persona_profile = {key: val for key, val in self.meta_persona_profile.items() if key != 'persona_hub'}


    def update_thought(self, therapist_utterance: str) -> str:
        """
        Update the agent's thought given the therapist's utterance
        """
        raise NotImplementedError()


    def utter(
        self, 
        self_utterance: str = '',
        therapist_utterance: str = ''
    ) -> str:

        assert self.persona_profile, \
            "Persona profile is not set. Please set it using 'set_persona(persona_profile)' before calling the utter method."

        return self.update_thought(therapist_utterance=therapist_utterance)
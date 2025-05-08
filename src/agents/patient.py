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
from src.utils.vllm_inference_utils import vLLMOffline, OpenAIAsyncInference


class Patient(LMAgent):


    def __init__(
        self,
        vllm_client: vLLMOffline | None = None,
        openai_client: OpenAI | None = None,
        openai_async_client: AsyncOpenAI | OpenAIAsyncInference | None = None,
    ) -> None:

        assert (openai_client is not None) or (openai_async_client is not None), \
            "Either openai_client or openai_async_client must be provided"

        super().__init__(
            client=openai_client, 
            async_client=openai_async_client,
            vllm_client=vllm_client,
        )

        self.persona_profile = ''
        self.update_thought_template = yaml.safe_load(
            open('./prompts/update_thought.yaml')
        )


    @property
    def persona(self) -> str:
        """
        Get the persona profile for the agent
        """
        return self.persona_profile


    def set_persona(self, persona_profile: str) -> None:
        """
        Set the persona profile for the agent
        """
        self.persona_profile = persona_profile


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
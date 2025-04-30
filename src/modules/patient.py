"""
Patient Agent. Frozen during RL training.
"""
import yaml
from openai import OpenAI, AsyncOpenAI

from utils.base_agent import LMAgent


class Patient(LMAgent):


    def __init__(
        self,
        client: OpenAI,
        async_client: AsyncOpenAI,
    ) -> None:
        super().__init__(
            client=client, 
            async_client=async_client
        )

        self.persona_profile = ''
        self.initial_thought_template = yaml.safe_load(
            open('./prompts/initial_thought.yaml')
        )
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
        
        # TODO: update the system message for the templates


    def update_thought(self, therapist_utterance: str) -> str:
        """
        Update the agent's thought given the therapist's utterance
        """
        raise NotImplementedError()


    def produce_initial_thought(self, self_utterance: str) -> str:
        """
        Produce the initial thought given the agent's own utterance that describes a situation
        """
        raise NotImplementedError()


    def utter(
        self, 
        self_utterance: str = '',
        therapist_utterance: str = ''
    ) -> str:

        assert self.persona_profile, \
            "Persona profile is not set. Please set it using 'set_persona(persona_profile)' before calling the utter method."

        if self_utterance and not therapist_utterance:
            return self.produce_initial_thought(self_utterance=self.persona_profile)
        else:
            return self.update_thought(therapist_utterance=therapist_utterance)

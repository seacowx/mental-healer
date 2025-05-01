"""
Patient Agent. Frozen during RL training.
"""
import yaml
from copy import deepcopy

from openai import OpenAI, AsyncOpenAI

from utils.base_agent import LMAgent
from utils.llm_inference import vLLMOffline


class Patient(LMAgent):


    def __init__(
        self,
        vllm_client: vLLMOffline = None,
        openai_client: OpenAI = None,
        openai_async_client: AsyncOpenAI = None,
    ) -> None:
        super().__init__(
            client=openai_client, 
            async_client=openai_async_client,
            vllm_client=vllm_client,
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


    def produce_initial_thought(
        self, 
        data: dict,
    ) -> list:
        """
        Produce the initial thought given the agent's own utterance that describes a situation
        This process is done in batches.

        Inputs:
            self_utterance_list (list): A list of agents' own utterances that describe a situation
            self_persona_list (list): A list of agents' own personas that describe a situation

        Outputs:
            initial_thought_list (list): A list of initial thoughts produced by the agent
        """

        initial_thought_message_list = []
        for key, val in data.items():

            cur_situation = val['situation']
            cur_persona = val['persona_profile']

            cur_prompt = deepcopy(self.initial_thought_template)

            system_content = cur_prompt['system'].replace('{{persona_profile}}', cur_persona)
            user_content = cur_prompt['user'].replace('{{persona_profile}}', cur_persona) \
                .replace('{{situation}}', cur_situation)

            cur_message = [
                {'role': 'system', 'content': system_content},
                {'role': 'user', 'content': user_content},
            ]

            initial_thought_message_list.append(cur_message)

        # get the initial thought
        # TODO: add flag for allowing thinking or not 
        output = self.vllm_client.inference(
            message_list=initial_thought_message_list[:10],
        )

        print(output)
        raise SystemExit()


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

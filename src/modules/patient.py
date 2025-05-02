"""
Patient Agent. Frozen during RL training.
"""
import yaml, json
import pandas as pd
from copy import deepcopy

from openai import OpenAI, AsyncOpenAI

from utils.base_agent import LMAgent
from utils.llm_inference import vLLMOffline
from modules.therapist_reward import TherapistReward
from utils.thought_utils import iterative_thought_generation


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


    def update_thought(self, therapist_utterance: str) -> str:
        """
        Update the agent's thought given the therapist's utterance
        """
        raise NotImplementedError()


    def produce_initial_thought(
        self, 
        data: dict,
        therapist_reward: TherapistReward,
        enable_thinking: bool = True,
    ) -> list:
        """
        Produce the initial thought given the agent's own utterance that describes a situation
        This process is done in batches.

        The generation process is iterative and follows these steps:
            1. Given a situation and persona profile, the agent generates an initial thought.
            2. The situation and the initial thought are passed to the sentiment reward model to get a sentiment result.
            3. The initial thought is valid if it results in negative sentiment. Otherwise, regenerate the thought.
            4. Steps 1-3 are repeated until all the initial thoughts result in negative sentiment.

        Inputs:
            self_utterance_list (list): A list of agents' own utterances that describe a situation
            self_persona_list (list): A list of agents' own personas that describe a situation

        Outputs:
            initial_thought_list (list): A list of initial thoughts produced by the agent
        """

        initial_thought_message_list = []
        situation_list = []
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

            situation_list.append(cur_situation)
            initial_thought_message_list.append(cur_message)

        initial_thought_message_list = initial_thought_message_list
        queue_idx_list = list(range(len(initial_thought_message_list)))
        TOLERANCE = 5

        # TODO: finish implementing the iterative thought generating process

        parsed_initial_thought_list = iterative_thought_generation(
            initial_thought_message_list=initial_thought_message_list,
            situation_list=situation_list,
            therapist_reward=therapist_reward,
            vllm_client=self.vllm_client,
            queue_idx_list=queue_idx_list,
            TOLERANCE=TOLERANCE,
        )

        situation_list = [
            val['situation'] for val in data.values()
        ]
        persona_list = [
            val['persona_profile'] for val in data.values()
        ]

        out_data = {}
        for initial_thought, (key, val) in zip(parsed_initial_thought_list, data.items()):
            out_data[key] = {
                'situation': val['situation'],
                'persona_profile': val['persona_profile'],
                'initial_thought': initial_thought,
            }

        with open('../data/situations/situations_with_initial_thought.json', 'w') as f:
            json.dump(out_data, f, indent=4)



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

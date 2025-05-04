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
from utils.llm_inference_utils import vLLMOffline
from rewards.therapist_reward import TherapistReward
from utils.thought_utils import iterative_thought_generation


class Patient(LMAgent):


    def __init__(
        self,
        vllm_client: vLLMOffline | None = None,
        openai_client: OpenAI | None = None,
        openai_async_client: AsyncOpenAI | None = None,
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
        disable_thinking: bool = False,
        regenerate_thought: bool = False,
    ) -> list:
        """
        Produce the initial thought given the agent's own utterance that describes a situation
        This process is done in batches.

        Inputs:
            data (dict): A dictionary containing the situation and persona profile
            therapist_reward (TherapistReward): The reward model for sentiment analysis
            disable_thinking (bool): Whether to disable reasoning mode when producing initial thoughts
            regenerate_thought (bool): Whether to regenerate the initial thought

        Outputs:
            initial_thought_list (list): A list of initial thoughts produced by the agent
        """

        # avoid re-generating the initial thought if it already exists
        cache_fpath = '../data/situations/situations_with_initial_thought.json'
        # if os.path.exists(cache_fpath) and not regenerate_thought:
        #     out_data = json.load(open(cache_fpath, 'r'))
        #     parsed_initial_thought_list = [
        #         val['initial_thought'] for val in out_data.values()
        #     ]
        #     return parsed_initial_thought_list

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
        TOLERANCE = 5

        parsed_initial_thought_list = iterative_thought_generation(
            initial_thought_message_list=initial_thought_message_list,
            situation_list=situation_list,
            therapist_reward=therapist_reward,
            vllm_client=self.vllm_client,
            enable_thinking=operator.not_(disable_thinking),
            TOLERANCE=TOLERANCE,
        )

        out_data = {}
        num_invalid_thought = 0
        for initial_thought, (key, val) in zip(parsed_initial_thought_list, data.items()):

            # check if the initial thought is valid, invalid thoughts are represented as empty strings
            if initial_thought:
                out_data[key] = {
                    'situation': val['situation'],
                    'persona_profile': val['persona_profile'],
                    'initial_thought': initial_thought,
                }
            else:
                num_invalid_thought += 1

        print(f"Number of invalid initial thoughts: {num_invalid_thought}")

        with open('../data/situations/situations_with_initial_thought.json', 'w') as f:
            json.dump(out_data, f, indent=4)

        return parsed_initial_thought_list


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
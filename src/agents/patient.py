"""
Patient Agent. Frozen during RL training.

This agent is responsible for generating the initial thought and updating the thought based on the therapist's utterance.
"""

import yaml, json
from jinja2 import Template

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
        
        # organize patient prompt templates
        self.patient_template = yaml.safe_load(
            open(patient_template_path)
        )
        self.patient_reaction_system = self.patient_template['react_to_therapist_utterance']['system']
        self.patient_reaction_user = Template(self.patient_template['react_to_therapist_utterance']['user'])


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


    def _make_patient_new_thought_msg(
        self,
        situation_desc_list: list[str],
        session_buffer: TherapeuticSessionBuffer,
        active_sample_idx_list: list[int],
        active_coping_strategy_idx_list: list[list[int]],
    ) -> list[dict[str, str]]:

        patient_new_thought_msg_list = []
        for sample_idx in range(len(situation_desc_list)):

            if sample_idx not in active_sample_idx_list:
                patient_new_thought_msg_list.append({})

            cur_persona_profile = self.meta_persona_profile[sample_idx]

            # retrieve the dialogue history corresponding to the current sample index
            cur_dialogue_history = session_buffer.get_dialogue_history(sample_idx=sample_idx)
            cur_thought = session_buffer.get_thought_history(sample_idx=sample_idx)

            cur_persona_profile_desc = verbalize_persona_profile(
                persona_profile_dict=cur_persona_profile
            )

            cur_situation_desc = situation_desc_list[sample_idx]

            # make a prompt for each of the coping strategies. 
            # the only thing that changes by coping strategy is the therapist's utterance (therapist_utterance)
            for coping_dialogue_list in cur_dialogue_history.values():
                role, therapist_utterance = coping_dialogue_list[-1].values()

                # ensure that the last utterance is from the therapist
                if role != 'therapist':
                    patient_new_thought_msg_list.append([])
                else:
                    patient_new_thought_msg = [
                        {'role': 'system', 'content': self.patient_reaction_system},
                        {'role': 'user', 'content': self.patient_reaction_user.render(
                                persona_profile=cur_persona_profile_desc,
                                situation=cur_situation_desc,
                                thought=cur_thought,
                                therapist_utterance=therapist_utterance,
                            )
                        }
                    ]

                patient_new_thought_msg_list.append(patient_new_thought_msg)

        return patient_new_thought_msg_list


    def utter(
        self, 
        situation_desc_list: list[str],
        patient_thought_list: list[str],
        session_buffer: TherapeuticSessionBuffer,
        active_sample_idx_list: list[int],
        active_coping_strategy_idx_list: list[list[int]],
    ) -> str:

        assert self.meta_persona_profile, \
            (
                "Persona profile is not set. Please set it using " 
                "'set_persona(persona_profile_dict_list)' before calling the utter method."
            )

        patient_new_thought_msg_list = self._make_patient_new_thought_msg(
            situation_desc_list=situation_desc_list,
            session_buffer=session_buffer,
            active_sample_idx_list=active_sample_idx_list,
            active_coping_strategy_idx_list=active_coping_strategy_idx_list,
        )

        print(patient_new_thought_msg_list)
        raise SystemExit

        new_thought_list = self.base_vllm_model.inference(
            message_list=patient_new_thought_msg_list,
        )

        parsed_response_list = []
        for response in new_thought_list:
            # parse the response, only retain the utterance
            if '<updated_thought>' in response:
                response = response.rsplit('<updated_thought>', 1)[1].split('</updated_thought>')[0].strip()
            parsed_response_list.append(response)

        # group the parsed response by coping strategy (list of list where the inner list contains 8 responses)
        out_response_list = []
        prev_active_coping_strategies = 0
        for sample_idx in range(len(situation_desc_list)):
            cur_active_coping_strategies = session_buffer.get_number_of_active_coping_strategies(
                sample_idx=sample_idx
            )

            out_response_list.append(
                parsed_response_list[prev_active_coping_strategies:cur_active_coping_strategies]
            )
            prev_active_coping_strategies += cur_active_coping_strategies

        print(out_response_list)
        raise SystemExit

        return out_response_list


                

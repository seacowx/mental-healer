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

        patient_new_thought_msg_list = []
        for sample_idx in range(len(situation_desc_list)):

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
            # TODO: consider whether to add a sentiment checker here: stop producing prompt if the sentiment has converted to positive
            for coping_strategy, coping_dialogue_list in cur_dialogue_history.items():
                role, therapist_utterance = coping_dialogue_list[-1].values()

                # ensure that the last utterance is from the therapist
                assert role == 'therapist', \
                    (
                        "The last utterance in the dialogue history should be from the therapist. "
                        f"Instead, the last utterance is from the {role}."
                    )

                patient_new_thought_msg = [
                    {'role': 'system', 'content': self.patient_reaction_system},
                    {
                        'role': 'user', 
                        'content': self.patient_reaction_user.render(
                            persona_profile=cur_persona_profile_desc,
                            situation=cur_situation_desc,
                            thought=cur_thought,
                            therapist_utterance=therapist_utterance,
                        )
                    }
                ]

                print(patient_new_thought_msg)
                raise SystemExit

                patient_new_thought_msg_list.append(patient_new_thought_msg)

                

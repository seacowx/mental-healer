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
        coping_strategy_list: list[str],
        base_vllm_model: vLLMOffline | None = None,
        openai_client: OpenAI | None = None,
        openai_async_client: AsyncOpenAI | OpenAIAsyncInference | None = None,
        patient_template_path: str = './prompts/patient.yaml',
    ) -> None:

        self.meta_persona_profile = []
        self.role_playing_instruction_list = []
        self.coping_strategy_list = coping_strategy_list

        # organize patient prompt templates
        self.patient_template = yaml.safe_load(
            open(patient_template_path)
        )
        self.patient_reaction_system = self.patient_template['react_to_therapist_utterance']['system']
        self.patient_reaction_user = Template(self.patient_template['react_to_therapist_utterance']['user'])

        super().__init__(
            openai_client=openai_client, 
            openai_async_client=openai_async_client,
            base_vllm_model=base_vllm_model,
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


    def _make_patient_new_thought_msg(
        self,
        turn_idx: int,
        situation_desc_list: list[str],
        session_buffer: TherapeuticSessionBuffer,
        active_sample_idx_list: list[int],
    ) -> tuple[list[dict[str, str]], list[str]]:

        patient_new_thought_msg_list = []
        sample_and_strategy_idx_list = []
        for sample_idx in range(len(situation_desc_list)):

            if sample_idx not in active_sample_idx_list:
                patient_new_thought_msg_list.append([])

            cur_persona_profile = self.meta_persona_profile[sample_idx]

            # retrieve the dialogue buffer corresponding to the current sample index
            cur_dialogue = session_buffer.get_dialogue(sample_idx=sample_idx)
            cur_thought = session_buffer.get_thought(turn_idx=turn_idx, sample_idx=sample_idx)

            cur_persona_profile_desc = verbalize_persona_profile(
                persona_profile_dict=cur_persona_profile
            )

            cur_situation_desc = situation_desc_list[sample_idx]

            # make a prompt for each of the coping strategies. 
            # the only thing that changes by coping strategy is the therapist's utterance (therapist_utterance)
            for coping_strategy_idx, coping_dialogue_list in enumerate(cur_dialogue.values()):

                # if the dialogue buffer is empty, skip
                if not coping_dialogue_list:
                    patient_new_thought_msg_list.append([])
                    continue

                role, therapist_utterance = coping_dialogue_list[-1].values()

                # ensure that the last utterance is from the therapist
                if role != 'therapist':
                    patient_new_thought_msg_list.append([])
                else:
                    patient_new_thought_msg_list.append([
                        {'role': 'system', 'content': self.patient_reaction_system},
                        {'role': 'user', 'content': self.patient_reaction_user.render(
                                persona_profile=cur_persona_profile_desc,
                                situation=cur_situation_desc,
                                previous_thought=cur_thought[coping_strategy_idx],
                                therapist_utterance=therapist_utterance,
                            )
                        }
                    ])

                    sample_and_strategy_idx_list.append((sample_idx, coping_strategy_idx))

        return patient_new_thought_msg_list, sample_and_strategy_idx_list


    def utter(
        self, 
        turn_idx: int,
        situation_desc_list: list[str],
        session_buffer: TherapeuticSessionBuffer,
        active_sample_idx_list: list[int],
        show_vllm_tqdm_bar: bool = False,
    ) -> tuple[list[dict[str, str]], list[str]]:
        """
        Generate the patient's new thought

        Outputs:
            parsed_response_list: list[dict[str, str]]: list of dicts, each containing the coping strategy name and the patient's new thought
            updated_patient_thought_list: list[str]: list of strings, each containing the patient's new thought. This is used to update the patient's thought buffer.
        """

        assert self.meta_persona_profile, \
            (
                "Persona profile is not set. Please set it using " 
                "'set_persona(persona_profile_dict_list)' before calling the utter method."
            )

        patient_new_thought_msg_list, sample_and_strategy_idx_list = self._make_patient_new_thought_msg(
            turn_idx=turn_idx,
            situation_desc_list=situation_desc_list,
            session_buffer=session_buffer,
            active_sample_idx_list=active_sample_idx_list,
        )

        # get rid of the completed coping strategies
        patient_new_thought_msg_list = [ele for ele in patient_new_thought_msg_list if ele]

        new_thought_list = self.base_vllm_model.inference(
            message_list=patient_new_thought_msg_list,
            show_tqdm_bar=show_vllm_tqdm_bar,
        )

        print(new_thought_list)
        raise SystemExit

        updated_patient_thought_list = [
            [''] * len(self.coping_strategy_list)
            for _ in range(max(active_sample_idx_list)+1)
        ]
        parsed_response_list = []
        for response_idx, response in enumerate(new_thought_list):
            cur_sample_idx, cur_strategy_idx = sample_and_strategy_idx_list[response_idx]

            cur_strategy_name = self.coping_strategy_list[cur_strategy_idx]

            # parse the response, only retain the utterance
            if '<updated_thought>' in response:
                response = response.rsplit('<updated_thought>', 1)[1].split('</updated_thought>')[0].strip()

            parsed_response_list.append({
                'coping_strategy': str(cur_sample_idx) + '||' + cur_strategy_name,
                'response': response,
            })
            updated_patient_thought_list[cur_sample_idx][cur_strategy_idx] = response

        return parsed_response_list, updated_patient_thought_list
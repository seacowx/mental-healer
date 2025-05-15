import yaml
import copy
from typing import Optional

from agents.planner import CopingAgent
from agents.patient import PatientAgent
from agents.therapist import TherapistAgent
from utils.vllm_inference_utils import vLLMOffline
from utils.therapeutic_utils import TherapeuticSessionBuffer


class TherapeuticSession:

    def __init__(
        self,
        base_vllm_model: vLLMOffline,
        coping_agent: Optional[CopingAgent] = None,
        coping_cot_templates_path: str = './prompts/coping_strategies.yaml',
        patient_prompt_template_path: str = './prompts/update_thought.yaml',
        max_turns: int = 5,
    ):
        self.therapist_agent = TherapistAgent(
            base_vllm_model=base_vllm_model,
        )
        self.patient_agent = PatientAgent(
            base_vllm_model=base_vllm_model,
            patient_template_path=patient_prompt_template_path,
        )
        self.coping_agent = coping_agent
        self.max_turns = max_turns

        self.base_vllm_model = base_vllm_model
        self.patient_prompt_template_path = patient_prompt_template_path
        self.coping_cot_templates = yaml.safe_load(open(coping_cot_templates_path))
        self.patient_prompt_template = yaml.safe_load(open(patient_prompt_template_path))

        self.patient_thought_update_template = self.patient_prompt_template['react_to_therapist_utterance']


    # TODO: add support for multiple samples batched inference
    def batch_simulate_therapeutic_session(
        self, 
        data: dict[str, dict],
        batch_size: int = 1,
    ):

        situation_key_list = list(data.keys())
        situation_dict_list = [data[key] for key in situation_key_list]
    
        # batch situation_dict_list according to the number of samples
        situation_dict_list_batches = [
            situation_dict_list[i:i+batch_size]
            for i in range(0, len(situation_dict_list), batch_size)
        ]
        session_buffer = TherapeuticSessionBuffer(batch_size=batch_size)

        for situation_dict_batch in situation_dict_list_batches:

            cur_situation_list = [ele['situation'] for ele in situation_dict_batch]
            cur_thought_list = [ele['initial_thought'] for ele in situation_dict_batch]
            cur_persona_profile_list = [ele['persona_profile'] for ele in situation_dict_batch]

            self.patient_agent.set_persona(
                persona_profile_dict_list=cur_persona_profile_list
            )

            # instantiate a patient agent and set the persona profile
            for _ in range(self.max_turns):
                # generate the therapist's utterance
                therapist_utterance_dict_list = self.therapist_agent.utter(
                    situation_desc_list=cur_situation_list,
                    patient_thought_list=cur_thought_list,
                    patient_persona_profile_list=cur_persona_profile_list,
                    session_buffer=session_buffer,
                )

                # update the session history
                for therapist_utterance_dict in therapist_utterance_dict_list:
                    utterance_idx, coping_strategy = therapist_utterance_dict['coping_strategy'].split('||')
                    utterance_idx = int(utterance_idx)
                    coping_utterance = therapist_utterance_dict['response']

                    session_buffer.update_buffer(
                        role='therapist',
                        sample_idx=utterance_idx,
                        coping_strategy=coping_strategy,
                        coping_utterance=coping_utterance,
                        thought=cur_thought_list[utterance_idx],
                    )

                # TODO: finish implementing this: patient agent should react to the therapist's utterance by producing a new thought
                # generate the patient's new thought
                patient_new_thought_list = self.patient_agent.utter(
                    situation_desc_list=cur_situation_list,
                    patient_thought_list=cur_thought_list,
                    session_buffer=session_buffer,
                )

                # # generate the patient's new thought
                # patient_new_thought = self.patient_agent.generate_new_thought(therapist_utterance)


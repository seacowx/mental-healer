import yaml
import copy
from typing import Optional

from agents.planner import CopingAgent
from agents.patient import PatientAgent
from agents.therapist import TherapistAgent
from utils.therapeutic_utils import SessionHistory
from utils.vllm_inference_utils import vLLMOffline


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
        self.coping_agent = coping_agent
        self.max_turns = max_turns

        self.base_vllm_model = base_vllm_model
        self.patient_prompt_template_path = patient_prompt_template_path
        self.coping_cot_templates = yaml.safe_load(open(coping_cot_templates_path))
        self.patient_prompt_template = yaml.safe_load(open(patient_prompt_template_path))

        self.patient_thought_update_template = self.patient_prompt_template['react_to_therapist_utterance']


    def simulate_therapeutic_session(self, situation_dict_list: list[dict]):

        for situation_dict in situation_dict_list:

            cur_situation = situation_dict['situation']
            cur_thought = situation_dict['initial_thought']
            cur_persona_profile = situation_dict['persona_profile']

            # instantiate a patient agent and set the persona profile
            cur_patient_agent = PatientAgent(
                base_vllm_model=self.base_vllm_model,
                patient_template_path=self.patient_prompt_template_path,
            )
            cur_patient_agent.set_persona(cur_persona_profile)

            # start the therapeutic session
            session_history = SessionHistory()
            for _ in range(self.max_turns):
                # generate the therapist's utterance
                therapist_utterance_dict_list = self.therapist_agent.utter(
                    situation_desc_list=[cur_situation],
                    patient_thought_list=[cur_thought],
                    patient_persona_profile_list=[cur_persona_profile],
                    session_history=session_history,
                )

                print(therapist_utterance_dict_list)
                raise SystemExit

                # update the session history
                session_history.add_utterance(
                    sample_idx_list=[0],
                    coping_strategy_list=[],
                    utterance_list=therapist_utterance_list,
                    role_list=['therapist'],
                )

                # TODO: finish implementing this: patient agent should react to the therapist's utterance by producing a new thought
                # # generate the patient's new thought
                patient_new_thought_list = cur_patient_agent.utter(
                    situation_desc_list=[cur_situation],
                    patient_thought_list=[cur_thought],
                    patient_persona_profile_list=[cur_persona_profile],
                    session_history=session_history,
                )

                # # generate the patient's new thought
                # patient_new_thought = self.patient_agent.generate_new_thought(therapist_utterance)
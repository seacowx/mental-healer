import yaml
from typing import Optional

from agents.planner import CopingAgent
from agents.patient import PatientAgent
from agents.therapist import TherapistAgent


class SessionHistory:

    def __init__(self,):
        self.coping_strategies = [
            'meta_rp_commit',
            'meta_rc_commit',
            'object_rp_commit',
            'object_rc_commit',
            'meta_rp_decommit',
            'meta_rc_decommit',
            'object_rp_decommit',
            'object_rc_decommit',
        ]

        self.coping_strategies_history = {
            coping_strategy: [] for coping_strategy in self.coping_strategies
        }

    
    def add_coping_strategy(self, coping_strategy: str, utterance: str, role: str):
        self.coping_strategies_history[coping_strategy].append({
            'role': role,
            'utterance': utterance,
        })

    
    @property
    def show_coping_strategies_history(self) -> dict:
        return self.coping_strategies_history


    def reset(self,):
        self.coping_strategies_history = {
            coping_strategy: [] for coping_strategy in self.coping_strategies
        }


class TherapeuticSession:

    def __init__(
        self,
        therapist_agent: TherapistAgent,
        patient_agent: PatientAgent,
        coping_agent: Optional[CopingAgent] = None,
        coping_cot_templates_path: str = './prompts/coping_strategies.yaml',
        max_turns: int = 5,
    ):
        self.therapist_agent = therapist_agent
        self.patient_agent = patient_agent
        self.coping_agent = coping_agent

        self.coping_cot_templates = yaml.safe_load(open(coping_cot_templates_path))


    def simulate_therapeutic_session(self, situation_dict: dict):

        cur_situation = situation_dict['situation']
        cur_thought = situation_dict['initial_thought']
        cur_persona_profile = situation_dict['persona_profile']

        # set the persona profile for the patient agent
        self.patient_agent.set_persona(cur_persona_profile)

        # start the therapeutic session
        session_history = SessionHistory()
        for _ in range(self.max_turns):
            # generate the therapist's utterance
            therapist_utterance = self.therapist_agent.utter(
                situation_desc=cur_situation,
                patient_thought=cur_thought,
                patient_persona_profile=cur_persona_profile,
                session_history=session_history,
            )

            # # generate the patient's new thought
            # patient_new_thought = self.patient_agent.generate_new_thought(therapist_utterance)

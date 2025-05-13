import yaml
from typing import Optional

from agents.planner import CopingAgent
from agents.patient import PatientAgent
from agents.therapist import TherapistAgent


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


    def sample_therapist_utterances(self,):
        ...


    def generate_patient_new_thought(self,):
        ...


    def generate_therapist_new_utterance(self,):
        ...


    def simulate_therapeutic_session(self, situation_dict: dict):

        cur_situation = situation_dict['situation']
        cur_thought = situation_dict['initial_thought']
        cur_persona_profile = situation_dict['persona_profile']

        self.patient_agent.set_persona(cur_persona_profile)
        self.patient_agent.persona_profile
        raise SystemExit()
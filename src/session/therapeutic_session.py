import yaml
from typing import Optional

from agents.planner import CopingAgent
from agents.patient import PatientAgent
from agents.therapist import TherapistAgent
from utils.therapeutic_utils import SessionHistory


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
        self.max_turns = max_turns

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
                situation_desc_list=[cur_situation],
                patient_thought_list=[cur_thought],
                patient_persona_profile_list=[cur_persona_profile],
                session_history=session_history,
            )

            # # generate the patient's new thought
            # patient_new_thought = self.patient_agent.generate_new_thought(therapist_utterance)
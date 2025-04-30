"""
Patient Agent. Frozen during RL training.
"""
from utils.base_agent import LMAgent

class Patient(LMAgent):


    def __init__(
        self,
        personality_trait: tuple,
    ) -> None:
        super().__init__()

        self.presonality_trait = personality_trait


    def init_model(self) -> None:
        """
        Initialize the agent
        """
        self.init_model()


    def terminate_model(self) -> None:
        """
        Terminate the agent
        """
        self.terminate_model()


    def update_thought(self, therapist_utterance: str) -> str:
        """
        Update the agent's thought given the therapist's utterance
        """
        raise NotImplementedError()


    def produce_initial_thought(self, self_utterance: str) -> str:
        """
        Produce the initial thought given the agent's own utterance that describes a situation
        """
        raise NotImplementedError()

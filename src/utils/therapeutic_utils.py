import yaml
from typing import Optional


class TherapeuticSessionBuffer:

    def __init__(self, batch_size: int = 8):
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
        self.batch_size = batch_size

        # sentiment buffer stores the sentiment after each turn of the therapeutic session
        self.sentiment_buffer = [[] for _ in range(self.batch_size)]
        # coping strategies history stores the complete dialogue history of each coping strategy of each sample
        self.session_history = [{
            coping_strategy: [] for coping_strategy in self.coping_strategies
        } for _ in range(self.batch_size)]

    
    def add_utterance(
        self, 
        role: str,
        sample_idx: int,
        coping_strategy: str,
        coping_utterance: str,
    ):
        self.session_history[sample_idx][coping_strategy].append({
            'role': role,
            'utterance': coping_utterance,
        })

    
    @property
    def current_session_history(self) -> dict:
        return self.session_history


    def reset(self,):
        self.session_history = [{
            coping_strategy: [] for coping_strategy in self.coping_strategies
        } for _ in range(self.batch_size)]
        self.sentiment_buffer = [[] for _ in range(self.batch_size)]
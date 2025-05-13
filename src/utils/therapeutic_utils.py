import yaml
from typing import Optional



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
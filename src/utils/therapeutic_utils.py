import yaml
from typing import Optional


class SessionHistory:

    def __init__(self, n_samples: int = 8):
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
        self.n_samples = n_samples

        # sentiment buffer stores the sentiment after each turn of the therapeutic session
        self.sentiment_buffer = [[] for _ in range(self.n_samples)]
        # coping strategies history stores the complete dialogue history of each coping strategy of each sample
        self.coping_strategies_history = [{
            coping_strategy: [] for coping_strategy in self.coping_strategies
        } for _ in range(self.n_samples)]

    
    def add_utterance(
        self, 
        sample_idx_list: list[int],
        coping_strategy_list: list[str], 
        therapist_utterance_dict_list: list[dict] = [],
        patient_utterance_dict_list: list[dict] = [],
    ):

        if therapist_utterance_dict_list:
            role_list = ['therapist'] * len(therapist_utterance_dict_list)
        elif patient_utterance_dict_list:
            role_list = ['patient'] * len(patient_utterance_dict_list)
        else:
            raise ValueError('Either therapist_utterance_dict_list or patient_utterance_dict_list must be provided')

    
    @property
    def show_coping_strategies_history(self) -> dict:
        return self.coping_strategies_history


    def reset(self,):
        self.coping_strategies_history = [{
            coping_strategy: [] for coping_strategy in self.coping_strategies
        } for _ in range(self.n_samples)]
        self.sentiment_buffer = [[] for _ in range(self.n_samples)]
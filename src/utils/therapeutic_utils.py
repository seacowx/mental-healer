import yaml
from typing import Optional


class TherapeuticSessionBuffer:

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
        self.session_history = [{
            coping_strategy: [] for coping_strategy in self.coping_strategies
        } for _ in range(self.n_samples)]

    
    def add_utterance(
        self, 
        therapist_utterance_dict_list: list[dict] = [],
        patient_utterance_dict_list: list[dict] = [],
    ):

        if therapist_utterance_dict_list:
            role = 'therapist'
            utterance_dict_list = therapist_utterance_dict_list
        elif patient_utterance_dict_list:
            role = 'patient'
            utterance_dict_list = patient_utterance_dict_list
        else:
            raise ValueError('Either therapist_utterance_dict_list or patient_utterance_dict_list must be provided')

        for utterance_dict in utterance_dict_list:
            for idx_and_strategy, utterance in utterance_dict.items():
                sample_idx, coping_strategy = idx_and_strategy.split('||')
                self.session_history[int(sample_idx)][coping_strategy].append({
                    'role': role,
                    'utterance': utterance,
                })

    
    @property
    def session_history(self) -> dict:
        return self.session_history


    def reset(self,):
        self.session_history = [{
            coping_strategy: [] for coping_strategy in self.coping_strategies
        } for _ in range(self.n_samples)]
        self.sentiment_buffer = [[] for _ in range(self.n_samples)]
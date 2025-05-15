import json


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
        self.coping_strategy_history = [{
            coping_strategy: [] for coping_strategy in self.coping_strategies
        } for _ in range(self.batch_size)]
        self.thought_history = [[] for _ in range(self.batch_size)]

    
    def add_utterance(
        self, 
        role: str,
        sample_idx: int,
        coping_strategy: str,
        coping_utterance: str,
    ):
        self.coping_strategy_history[sample_idx][coping_strategy].append({
            'role': role,
            'utterance': coping_utterance,
        })


    def update_buffer(
        self, 
        role: str,
        sample_idx: int,
        coping_strategy: str,
        coping_utterance: str,
        thought: str,
    ):
        self.add_utterance(
            role=role,
            sample_idx=sample_idx,
            coping_strategy=coping_strategy,
            coping_utterance=coping_utterance,
        )
        self.thought_history[sample_idx] = thought
    
    @property
    def show_dialogue_history(self) -> dict:
        return json.dumps(self.coping_strategy_history)

    
    @property
    def show_thought_history(self) -> dict:
        return json.dumps(self.thought_history)


    def get_session_history(self, sample_idx: int) -> dict:
        return self.coping_strategy_history[sample_idx]

    
    def get_thought_history(self, sample_idx: int) -> dict:
        return self.thought_history[sample_idx]


    def reset(self,):
        self.coping_strategy_history = [{
            coping_strategy: [] for coping_strategy in self.coping_strategies
        } for _ in range(self.batch_size)]
        self.sentiment_buffer = [[] for _ in range(self.batch_size)]
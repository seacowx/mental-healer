import json, yaml


class TherapeuticSessionBuffer:

    def __init__(
        self, 
        coping_strategies_list: list[str],
        initial_thought_list: list[list[str]],
        batch_size: int = 8,
    ):

        self.batch_size = batch_size
        self.coping_strategy_list = coping_strategies_list
        self.coping_strategy_to_idx = {
            ele: idx for idx, ele in enumerate(self.coping_strategy_list)
        }

        # sentiment buffer stores the sentiment after each turn of the therapeutic session
        self.sentiment_buffer = [[] for _ in range(self.batch_size)]
        # coping strategies history stores the complete dialogue history of each coping strategy of each sample
        self.coping_dialogue_history = [{
            coping_strategy: [] for coping_strategy in self.coping_strategy_list
        } for _ in range(self.batch_size)]
        self.thought_history = {'0': initial_thought_list}
        
        # FIXME: remove this testing code
        # self.is_therapeutic_session_active = [[True] * len(self.coping_strategy_list)] * self.batch_size
        self.is_therapeutic_session_active = [[True, False, True, True, True, True, True, False]]

    
    def add_utterance(
        self, 
        role: str,
        sample_idx: int,
        coping_strategy: str,
        coping_utterance: str,
    ):
        self.coping_dialogue_history[sample_idx][coping_strategy].append({
            'role': role,
            'utterance': coping_utterance,
        })


    def update_buffer(
        self, 
        role: str,
        sample_idx: int,
        coping_strategy: str,
        coping_utterance: str,
        turn_idx: int | None = None,
        thought: list[str] | None = None,
    ):
        self.add_utterance(
            role=role,
            sample_idx=sample_idx,
            coping_strategy=coping_strategy,
            coping_utterance=coping_utterance,
        )

        # only update the thought history if turn_idx is provided: this is because the buffer is updated twice in each turn (once for the patient and once for the therapist) but the thought history is only updated once for each turn (after the patients' new thoughts are generated)
        if turn_idx:
            self.thought_history[str(turn_idx)] = thought


    def update_sentiment_buffer(
        self,
        sample_idx: int,
        coping_strategy: str,
        sentiment: str,
    ):
        coping_strategy_idx = self.coping_strategy_to_idx[coping_strategy]
        self.sentiment_buffer[sample_idx][coping_strategy_idx].append(sentiment)

        if sentiment == 'positive':
            self.is_therapeutic_session_complete[sample_idx][coping_strategy_idx] = True
    

    @property
    def show_dialogue_history(self) -> dict:
        return json.dumps(self.coping_dialogue_history)

    
    @property
    def show_thought_history(self) -> dict:
        return json.dumps(self.thought_history)


    def get_dialogue_history(self, sample_idx: int) -> dict:
        return self.coping_dialogue_history[sample_idx]

    
    def get_thought_history(self, sample_idx: int) -> dict:
        return self.thought_history[sample_idx]

    
    def get_latest_sentiment(self, coping_strategy_idx: int) -> str:
        if (temp_sentiment_list := self.sentiment_buffer[coping_strategy_idx]):
            return temp_sentiment_list[coping_strategy_idx][-1].lower()
        else:
            return 'negative'


    def get_session_status_list(self) -> list[int]:
        return self.is_therapeutic_session_active


    # def reset(self,):
    #     self.sentiment_buffer = [[] for _ in range(self.batch_size)]
    #     self.coping_dialogue_history = [
    #         {coping_strategy: [] for coping_strategy in self.coping_strategy_list } 
    #         for _ in range(self.batch_size)
    #     ]
    #     self.thought_history = {}
    #     self.is_therapeutic_session_complete = [False] * self.batch_size
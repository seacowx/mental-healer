import json, yaml


class TherapeuticSessionBuffer:

    def __init__(
        self, 
        batch_size: int,
        coping_strategies_list: list[str],
        initial_thought_list: list[list[str]],
    ):

        self.batch_size = batch_size
        self.coping_strategy_list = coping_strategies_list
        self.coping_strategy_to_idx = {
            ele: idx for idx, ele in enumerate(self.coping_strategy_list)
        }

        # sentiment buffer stores the sentiment after each turn of the therapeutic session
        self.sentiment_buffer = {}
        # coping strategies buffer stores the complete dialogue buffer of each coping strategy of each sample
        self.coping_dialogue_buffer = [{
            coping_strategy: [] for coping_strategy in self.coping_strategy_list
        } for _ in range(self.batch_size)]
        
        # turn_idx -> list of thoughts
        self.thought_buffer = {'0': initial_thought_list}
        
        self.is_therapeutic_session_active = [[True] * len(self.coping_strategy_list)] * self.batch_size
        # self.is_therapeutic_session_active = [[True, False, True, True, True, True, True, False]]

    
    def update_utterance_buffer(
        self, 
        role: str,
        sample_idx: int,
        coping_strategy: str,
        coping_utterance: str,
    ):
        self.coping_dialogue_buffer[sample_idx][coping_strategy].append({
            'role': role,
            'utterance': coping_utterance,
        })


    def update_thought_buffer(
        self,
        turn_idx: int,
        thought_list: list[str],
    ):
        self.thought_buffer[str(turn_idx)] = thought_list


    def update_sentiment_buffer(
        self,
        turn_idx: int,
        sentiment_list: list[list[str]],
    ):

        self.sentiment_buffer[str(turn_idx)] = sentiment_list

        # set the session status to complete if the sentiment is positive
        for sample_idx, sample_sentiment_list in enumerate(sentiment_list):
            for coping_strategy_idx, sentiment in enumerate(sample_sentiment_list):
                if sentiment == 'positive':
                    self.is_therapeutic_session_active[sample_idx][coping_strategy_idx] = False


    @property
    def show_dialogue_buffer(self) -> dict:
        return json.dumps(self.coping_dialogue_buffer)

    
    @property
    def show_thought_buffer(self) -> dict:
        return json.dumps(self.thought_buffer)

    
    @property
    def show_sentiment_buffer(self) -> dict:
        return json.dumps(self.sentiment_buffer)


    def get_dialogue(self, sample_idx: int) -> dict:
        return self.coping_dialogue_buffer[sample_idx]

    
    def get_thought(self, turn_idx: int, sample_idx: int) -> list[str]:
        return self.thought_buffer[str(turn_idx)][sample_idx]

    
    def get_latest_sentiment(self, coping_strategy_idx: int) -> str:
        if (temp_sentiment_list := self.sentiment_buffer[coping_strategy_idx]):
            return temp_sentiment_list[coping_strategy_idx][-1].lower()
        else:
            return 'negative'


    def get_session_status_list(self) -> list[int]:
        return self.is_therapeutic_session_active


    # def reset(self,):
    #     self.sentiment_buffer = [[] for _ in range(self.batch_size)]
    #     self.coping_dialogue_buffer = [
    #         {coping_strategy: [] for coping_strategy in self.coping_strategy_list } 
    #         for _ in range(self.batch_size)
    #     ]
    #     self.thought_buffer = {}
    #     self.is_therapeutic_session_complete = [False] * self.batch_size
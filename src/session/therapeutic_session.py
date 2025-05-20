import yaml
import copy
from typing import Optional

from agents.planner import CopingAgent
from agents.patient import PatientAgent
from agents.therapist import TherapistAgent
from rewards.sentiment import SentimentReward
from utils.vllm_inference_utils import vLLMOffline
from utils.therapeutic_utils import TherapeuticSessionBuffer


class TherapeuticSession:

    def __init__(
        self,
        base_vllm_model: vLLMOffline,
        coping_agent: Optional[CopingAgent] = None,
        coping_cot_templates_path: str = './prompts/coping_strategies.yaml',
        patient_prompt_template_path: str = './prompts/patient.yaml',
        coping_strategies_path: str = './configs/coping_strategy.yaml',
        sentiment_prompt_path: str = './prompts/sentiment.yaml',
        sentiment_reward_rule_path: str = './configs/sentiment_reward_rules.yaml',
        sentiment_mapping_path: str = './configs/emotion_to_sentiment.yaml',
        max_turns: int = 5,
    ):
        self.base_vllm_model = base_vllm_model
        self.patient_prompt_template_path = patient_prompt_template_path
        self.coping_cot_templates = yaml.safe_load(open(coping_cot_templates_path))
        self.patient_prompt_template = yaml.safe_load(open(patient_prompt_template_path))
        self.patient_thought_update_template = self.patient_prompt_template['react_to_therapist_utterance']
        self.coping_strategy_list = yaml.safe_load(open(coping_strategies_path, 'r'))

        self.therapist_agent = TherapistAgent(
            base_vllm_model=base_vllm_model,
        )
        self.patient_agent = PatientAgent(
            base_vllm_model=base_vllm_model,
            patient_template_path=patient_prompt_template_path,
            coping_strategy_list=self.coping_strategy_list,
        )
        self.sentiment_reward = SentimentReward(
            base_vllm_model=base_vllm_model,
            reward_rule_path = sentiment_reward_rule_path,
            sentiment_mapping_path = sentiment_mapping_path,
            sentiment_prompt_path=sentiment_prompt_path,
        )
        self.coping_agent = coping_agent
        self.max_turns = max_turns



    def _get_active_coping_strategy_list(self, session_buffer: TherapeuticSessionBuffer):

        # get the active coping strategies for each sample in the batch
        session_status_list = session_buffer.get_session_status_list()
        active_coping_strategy_idx_list = []
        for session_status in session_status_list:
            cur_active_idx_list = []
            for coping_idx, is_active in enumerate(session_status):
                if is_active:
                    cur_active_idx_list.append(coping_idx)

            active_coping_strategy_idx_list.append(cur_active_idx_list)

        return active_coping_strategy_idx_list


    def _update_session_buffer(
        self,
        session_buffer: TherapeuticSessionBuffer,
        turn_idx: int,
        utterance_dict_list: list[dict] = [],
        role: str = '',
        thought_list: list[list[str]] | None = None,
        sentiment_list: list[list[str]] | None = None,
    ) -> TherapeuticSessionBuffer:

        # the utterance buffer is updated for each utterance
        for utterance_dict in utterance_dict_list:
            utterance_idx, coping_strategy = utterance_dict['coping_strategy'].split('||')
            utterance_idx = int(utterance_idx)
            coping_utterance = utterance_dict['response']

            session_buffer.update_utterance_buffer(
                role=role,
                sample_idx=utterance_idx,
                coping_strategy=coping_strategy,
                coping_utterance=coping_utterance,
            )

        # update the thought buffer after new thoughts are generated, this is indicated by the `turn_idx`
        if thought_list:
            session_buffer.update_thought_buffer(
                turn_idx=turn_idx,
                thought_list=thought_list,
            )

        # TODO: implement the sentiment buffer update
        if sentiment_list:
            session_buffer.update_sentiment_buffer(
                sentiment_list=sentiment_list,
                turn_idx=turn_idx,
            )

        return session_buffer


    def _simulate_therapeutic_session(
        self,
        session_buffer: TherapeuticSessionBuffer,
        cur_situation_list: list[str],
        patient_thought_list: list[list[str]],
        cur_persona_profile_list: list[dict],
    ):

        for turn_idx in range(1, self.max_turns + 1):

            active_coping_strategy_idx_list = self._get_active_coping_strategy_list(
                session_buffer=session_buffer,
            )
            active_sample_idx_list = [
                sample_idx for sample_idx, active_coping_strategy_idx_list in enumerate(active_coping_strategy_idx_list)
                if active_coping_strategy_idx_list
            ]

            # generate the therapist's utterance
            therapist_utterance_dict_list = self.therapist_agent.utter(
                situation_desc_list=cur_situation_list,
                patient_thought_list=patient_thought_list,
                patient_persona_profile_list=cur_persona_profile_list,
                session_buffer=session_buffer,
                active_sample_idx_list=active_sample_idx_list,
                active_coping_strategy_idx_list=active_coping_strategy_idx_list,
            )

            # update the session buffer
            session_buffer = self._update_session_buffer(
                utterance_dict_list=therapist_utterance_dict_list,
                role='therapist',
                session_buffer=session_buffer,
                turn_idx=turn_idx,
            )

            # generate the patient's new thought and update `patient_thought_list`
            patient_thought_dict_list, patient_thought_list = self.patient_agent.utter(
                situation_desc_list=cur_situation_list,
                session_buffer=session_buffer,
                active_sample_idx_list=active_sample_idx_list,
            )

            # update the session buffer
            session_buffer = self._update_session_buffer(
                utterance_dict_list=patient_thought_dict_list,
                role='patient',
                session_buffer=session_buffer,
                thought_list=patient_thought_list,
                turn_idx=turn_idx,
            )

            patient_sentiment_list = self.sentiment_reward.get_sentiment(
                situation_desc_list=cur_situation_list,
                thought_list=patient_thought_list,
            )

            # for cur_thought_list, cur_sentiment_list in zip(patient_thought_list, patient_sentiment_list):
            #     for cur_thought, cur_sentiment in zip(cur_thought_list, cur_sentiment_list):
            #         print(cur_thought)
            #         print(cur_sentiment)
            #         print('-' * 100)

            session_buffer = self._update_session_buffer(
                session_buffer=session_buffer,
                sentiment_list=patient_sentiment_list,
                turn_idx=turn_idx,
            )

            print('-' * 100)
            print(session_buffer.show_dialogue_buffer)
            print('-' * 100)
            print(session_buffer.show_thought_buffer)
            print('-' * 100)
            print(session_buffer.show_sentiment_buffer)
            print('-' * 100)
            raise SystemExit


            # TODO: update sentiment to session buffer


    def batch_simulate_therapeutic_session(
        self, 
        data: dict[str, dict],
        batch_size: int = 1,
    ):

        situation_key_list = list(data.keys())
        situation_dict_list = [data[key] for key in situation_key_list]
    
        # batch situation_dict_list according to the number of samples
        situation_dict_list_batches = [
            situation_dict_list[i:i+batch_size]
            for i in range(0, len(situation_dict_list), batch_size)
        ]

        # iterate over each situation in the batch. For each situation, simulate the therapeutic session until
        # either the patient's thought is positive or the maximum number of turns is reached
        for situation_dict_batch in situation_dict_list_batches:

            cur_situation_list = [ele['situation'] for ele in situation_dict_batch]
            patient_thought_list = [
                [ele['initial_thought']] * len(self.coping_strategy_list) for ele in situation_dict_batch
            ]
            cur_persona_profile_list = [ele['persona_profile'] for ele in situation_dict_batch]

            # initialize the session buffer
            session_buffer = TherapeuticSessionBuffer(
                batch_size=batch_size,
                coping_strategies_list=self.coping_strategy_list,
                initial_thought_list=patient_thought_list,
            )

            # set the persona profile of the current patient batch
            self.patient_agent.set_persona(
                persona_profile_dict_list=cur_persona_profile_list
            )

            self._simulate_therapeutic_session(
                session_buffer=session_buffer,
                cur_situation_list=cur_situation_list,
                patient_thought_list=patient_thought_list,
                cur_persona_profile_list=cur_persona_profile_list,
            )


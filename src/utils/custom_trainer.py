"""
Custom GRPO Trainer

Methods that PROBABLY need to be modified:

- _get_per_token_logps: only compute logps for the tokens after </think> as everything before is inserted as tempate
- _prepare_inputs: insert the thought template in the input message
- _generate_and_score_completions: insert the thought template in the input message
"""

from trl import GRPOTrainer

from utils.optimizer_utils import get_grpo_optimizer, get_grpo_scheduler


class TherapeuticSession:

    def __init__(self,):
        ...

    def generate_patient_new_thought(self,):
        ...


    def generate_therapist_new_utterance(self,):
        ...


class CustomGRPOTrainer(GRPOTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 

        self.therapeutic_session = TherapeuticSession()

        print('\n\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
        self.model.print_trainable_parameters()
        print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n\n')


    def _reward_buffer(self,):
        """
        Stores the sentiment history of the current therapeutic session. 
        """
        ...
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
        self.peak_lr = kwargs.get('peak_learning_rate', 5.3e-4)
        self.adam_beta1 = kwargs.get('adam_beta1', 0.9)
        self.adam_beta2 = kwargs.get('adam_beta2', 0.95)
        self.weight_decay = kwargs.get('weight_decay', 0.1)
        self.warmup_steps = kwargs.get('warmup_steps', 2000)
        self.base_lr = kwargs.get('base_learning_rate', 0.)

        print('\n\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
        self.model.print_trainable_parameters()
        print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n\n')


    def _reward_buffer(self,):
        """
        Stores the sentiment history of the current therapeutic session. 
        """
        ...

    
    def create_optimizer_and_scheduler(
            self, 
            num_training_steps: int, 
            peak_learning_rate: float,
            adam_beta1: float,
            adam_beta2: float,
            weight_decay: float,
            warmup_steps: int,
            base_learning_rate: float,
        ):
        """
        Create the optimizer and scheduler for the model.
        """
        self.optimizer = get_grpo_optimizer(
            model=self.model,
            peak_learning_rate=peak_learning_rate,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            weight_decay=weight_decay,
        )
        self.lr_scheduler = get_grpo_scheduler(
            grpo_optimizer=self.optimizer,
            total_steps=num_training_steps,
            warmup_steps=warmup_steps,
            base_lr=base_learning_rate,
            peak_lr=peak_learning_rate,
        )


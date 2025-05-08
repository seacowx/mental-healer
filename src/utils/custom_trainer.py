"""
Custom GRPO Trainer

Methods that PROBABLY need to be modified:

- _get_per_token_logps: only compute logps for the tokens after </think> as everything before is inserted as tempate
- _prepare_inputs: insert the thought template in the input message
- _generate_and_score_completions: insert the thought template in the input message
"""

from trl import GRPOTrainer


class CustomGRPOTrainer(GRPOTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 
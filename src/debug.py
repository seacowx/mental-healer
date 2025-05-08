
"""
This is a minimal example to make sure that the original GRPO training works 
and the small custom edits are working properly.
"""

import yaml
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig


def main():
    dataset = load_dataset("trl-lib/tldr", split="train")

    def reward_func(completions, **kwargs):
        # Dummy reward function that rewards completions with more unique letters.
        return [float(len(set(completion))) for completion in completions]

    # define lora config
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=64,
        lora_alpha=256,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_rslora=True,
    )

    grpo_config_dict = yaml.safe_load(open('./configs/grpo.yaml', 'r'))
    grpo_config = GRPOConfig(**grpo_config_dict)

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=reward_func,
        train_dataset=dataset,
        peft_config=lora_config,
        args=grpo_config,
    )

    trainer.train()


if __name__ == "__main__":
    main()
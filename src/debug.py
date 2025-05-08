
"""
This is a minimal example to make sure that the original GRPO training works 
and the small custom edits are working properly.
"""

import os
import yaml
from peft import LoraConfig
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig

from utils.vllm_inference_utils import trlServer
from utils.custom_trainer import CustomGRPOTrainer


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

    TRL_VLLM_PORT = 8880
    grpo_config_dict = yaml.safe_load(open('./configs/grpo.yaml', 'r'))
    grpo_config_dict['vllm_server_port'] = TRL_VLLM_PORT
    grpo_config = GRPOConfig(**grpo_config_dict)

    # STEP: initialize trl vllm server
    trl_vllm_server = trlServer(
        model_path="Qwen/Qwen2-0.5B-Instruct",
        available_cuda_list=[1],
    )
    trl_vllm_server.start_trl_vllm_server(
        trl_vllm_port=TRL_VLLM_PORT,
    )

    # print('\n\n-----------------------------------------------------------------------')
    # print('Finished starting trl vllm server')
    # print('-----------------------------------------------------------------------\n\n')

    # trainer = GRPOTrainer(
    #     model="Qwen/Qwen2-0.5B-Instruct",
    #     reward_funcs=reward_func,
    #     train_dataset=dataset,
    #     peft_config=lora_config,
    #     args=grpo_config,
    # )

    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        trainer = CustomGRPOTrainer(
            model="Qwen/Qwen2-0.5B-Instruct",
            reward_funcs=reward_func,
            train_dataset=dataset,
            peft_config=lora_config,
            args=grpo_config,
        )

        print('\n\n-----------------------------------------------------------------------')
        print('Finished initializing custom grpo trainer. Training will start now.')
        print('-----------------------------------------------------------------------\n\n')

        trainer.train()
    finally:
        trl_vllm_server.kill_server()


if __name__ == "__main__":
    main()
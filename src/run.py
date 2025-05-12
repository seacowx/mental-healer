
"""
This is a minimal example to make sure that the original GRPO training works 
and the small custom edits are working properly.
"""

import os
import random
import argparse
import yaml, json
import numpy as np

import torch

from datasets import load_dataset
from trl import GRPOConfig

from utils.custom_trainer import CustomGRPOTrainer
from utils.data_utils import prepare_training_data
from utils.custom_trainer_args import GRPOTrainerArgs
from utils.model_utils import (
    initialize_models_and_agents,
    ensure_graceful_exit,
    ServerContainer,
)

from rewards.sentiment import SentimentReward
from rewards.therapist_reward import TherapistReward


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the RL training script.")
    parser.add_argument(
        '--therapist_base_model',
        type=str,
        default='Qwen/Qwen3-4B',
        help="The base model to use for the training.",
    )
    parser.add_argument(
        '--patient_base_model',
        type=str,
        default='Qwen/Qwen3-8B',
        help="The base model to use for the training.",
    )
    parser.add_argument(
        '--training_data_path',
        type=str,
        default='../data/situations/situations_with_initial_thought_top1.json',
        help="The path to the training data file. Default is '../data/situations/situations_with_initial_thought_top1.json'.",
    )
    parser.add_argument(
        '--grpo_config',
        type=str,
        default='./configs/grpo_config.yaml',
        help="The path to the training config file. Default is './configs/grpo.yaml'.",
    )
    parser.add_argument(
        "--lora_config",
        type=str,
        default='./configs/lora_config.yaml',
        help="The path to the LoRA config file. Default is './configs/lora_config.yaml'.",
    )
    parser.add_argument(
        "--trl_vllm_port", 
        type=int, 
        default=8880,
        help="The port to use for the trl vllm server."
    )
    return parser.parse_args()


def reward_func(completions, **kwargs):
    # Dummy reward function that rewards completions with more unique letters.
    return [float(len(set(completion))) for completion in completions]


# initialize the server container to keep track of the vllm servers
server_container = ServerContainer()

@ensure_graceful_exit(server_container)
def main():

    args = parse_args()

    # STEP: load training data, each instance contains the following fields:
    # situation (str): the situation description
    # initial_thought (str): the initial thought of the patient
    # persona (str): the persona of the patient
    # conversation_data, persona_data = prepare_training_data(
    #     data_path=args.training_data_path,
    # )

    # STEP: initialize agents. The patient agent uses the same LLM as the sentiment reward model. 
    agent_vllm_server = initialize_models_and_agents(
        patient_base_model=args.patient_base_model,
    ) 

    # patient_agent = initialize_patient_agent(
    #     patient_model=args.therapist_base_model,
    # )


    # FIXME: This dataset is for debugging only
    # dataset = load_dataset("trl-lib/tldr", split="train")


    # # STEP: load training config and lora config
    # grpo_config_dict = yaml.safe_load(open(args.grpo_config, 'r'))
    # grpo_config = GRPOTrainerArgs(**grpo_config_dict)
    # lora_config_dict = yaml.safe_load(open(args.lora_config, 'r'))

    # grpo_config_dict['vllm_server_port'] = args.trl_vllm_port

    # # define lora config and grpo config
    # lora_config = LoraConfig(
    #     **lora_config_dict
    # )

    # grpo_config = GRPOConfig(
    #     output_dir=grpo_config.output_dir,
    #     do_train=grpo_config.do_train,
    #     per_device_train_batch_size=grpo_config.per_device_train_batch_size,
    #     gradient_accumulation_steps=grpo_config.gradient_accumulation_steps,
    #     weight_decay=grpo_config.weight_decay,
    #     num_train_epochs=grpo_config.num_train_epochs,
    #     use_vllm=grpo_config.use_vllm,
    #     vllm_server_port=grpo_config.vllm_server_port,
    #     logging_steps=grpo_config.logging_steps,
    #     logging_first_step=grpo_config.logging_first_step,
    #     log_completions=grpo_config.log_completions,
    #     use_liger_kernel=grpo_config.use_liger_kernel,
    #     adam_beta1=grpo_config.adam_beta1,
    #     adam_beta2=grpo_config.adam_beta2,
    #     learning_rate=grpo_config.learning_rate,
    # )
    # grpo_trainer = CustomGRPOTrainer(
    #     model=args.therapist_base_model,
    #     reward_funcs=reward_func,
    #     train_dataset=dataset,
    #     peft_config=lora_config,
    #     args=grpo_config,
    # )

    # print('\n\n')
    # print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
    # print("GRPO Training Started...")
    # print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
    # print('\n\n')

    # grpo_trainer.train()


if __name__ == "__main__":
    main()
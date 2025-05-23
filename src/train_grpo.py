
"""
This is a minimal example to make sure that the original GRPO training works 
and the small custom edits are working properly.
"""

import os
import random
import asyncio
import argparse
import yaml, json
import numpy as np

import torch

from datasets import load_dataset
from trl import GRPOConfig

from utils.model_utils import load_all_models
from utils.custom_trainer import CustomGRPOTrainer
from utils.data_utils import prepare_training_data
from utils.custom_trainer_args import GRPOTrainerArgs

from agents.patient import PatientAgent
from rewards.sentiment import SentimentReward
from session.therapeutic_session import TherapeuticSession


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the RL training script.")
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
    return parser.parse_args()


def reward_func(completions, **kwargs):
    # Dummy reward function that rewards completions with more unique letters.
    return [float(len(set(completion))) for completion in completions]


def main():

    args = parse_args()

    # situation (str): the situation description
    # initial_thought (str): the initial thought of the patient
    # persona (str): the persona of the patient
    conversation_data, persona_data, situation_data = prepare_training_data(
        data_path=args.training_data_path,
    )

    # STEP: initialize agents. The patient agent uses the same LLM as the sentiment reward model. 
    grpo_config_dict = yaml.safe_load(open(args.grpo_config, 'r'))
    grpo_config = GRPOTrainerArgs(**grpo_config_dict)

    offline_vllm_base_model = load_all_models(
        base_model_path=grpo_config.base_agent_path,
        base_model_device=torch.device(grpo_config.base_model_device),
    )

    sentiment_reward_model = SentimentReward(
        base_vllm_model=offline_vllm_base_model,
    )

    patient_agent = PatientAgent(
        coping_strategy_config_path='./configs/coping_strategy.yaml',
        base_vllm_model=offline_vllm_base_model,
    )

    raise SystemExit

    # from debug import test_sentiment
    # test_sentiment(sentiment_reward_model=sentiment_reward_model)
    # raise SystemExit


    # dataset = load_dataset("trl-lib/tldr", split="train")


    # # STEP: load training config and lora config
    # lora_config_dict = yaml.safe_load(open(args.lora_config, 'r'))

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
    #     model=grpo_config.model_path,
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
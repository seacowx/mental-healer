
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
from torch.optim import AdamW

from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from utils.custom_trainer import CustomGRPOTrainer
from utils.data_utils import prepare_training_data
from utils.custom_trainer_args import GRPOTrainerArgs
from utils.agent_utils import initialize_patient_agent
from utils.optimizer_utils import get_grpo_optimizer_and_scheduler, compute_total_steps

from rewards.sentiment import SentimentReward
from rewards.therapist_reward import TherapistReward


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the RL training script.")
    parser.add_argument(
        '--base_model',
        type=str,
        default='Qwen/Qwen3-4B',
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


def main():

    args = parse_args()

    # STEP: load training data, each instance contains the following fields:
    # situation (str): the situation description
    # initial_thought (str): the initial thought of the patient
    # persona (str): the persona of the patient
    # conversation_data, persona_data = prepare_training_data(
    #     data_path=args.training_data_path,
    # )

    # STEP: initialize patient agent. The patient agent uses the same LLM as the therapist. 
    # patient_agent = initialize_patient_agent(
    #     patient_model=args.base_model,
    # )
    dataset = load_dataset("trl-lib/tldr", split="train")

    # STEP: load training config and lora config
    grpo_config_dict = yaml.safe_load(open(args.grpo_config, 'r'))
    grpo_config = GRPOTrainerArgs(**grpo_config_dict)
    lora_config_dict = yaml.safe_load(open(args.lora_config, 'r'))

    grpo_config_dict['vllm_server_port'] = args.trl_vllm_port

    # initialize the base model
    # it is easier to implement custom optimizer and scheduler this way
    base_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.base_model,
        torch_dtype="bfloat16",
    )
    # define lora config and grpo config
    lora_config = LoraConfig(
        **lora_config_dict
    )
    peft_model = get_peft_model(base_model, lora_config)

    # STEP: setup the optimizer and scheduler according to the original GRPO paper
    TOTAL_STEPS = compute_total_steps(
        num_train_epochs=grpo_config.num_train_epochs,
        per_device_train_batch_size=grpo_config.per_device_train_batch_size,
        gradient_accumulation_steps=grpo_config.gradient_accumulation_steps,
        len_dataset=len(dataset),
    )

    # return a tuple of AdamW optimizer and a step-wise scheduler
    prepared_optimizer = get_grpo_optimizer_and_scheduler(
        model=peft_model,
        total_steps=TOTAL_STEPS,
        adam_beta1=grpo_config.adam_beta1,
        adam_beta2=grpo_config.adam_beta2,
        weight_decay=grpo_config.weight_decay,
        warmup_steps=grpo_config.warmup_steps,
        base_lr=grpo_config.base_learning_rate,
        peak_lr=grpo_config.peak_learning_rate,
    )

    grpo_config = GRPOConfig(
        output_dir=grpo_config.output_dir,
        do_train=grpo_config.do_train,
        per_device_train_batch_size=grpo_config.per_device_train_batch_size,
        gradient_accumulation_steps=grpo_config.gradient_accumulation_steps,
        weight_decay=grpo_config.weight_decay,
        num_train_epochs=grpo_config.num_train_epochs,
        use_vllm=grpo_config.use_vllm,
        vllm_server_port=grpo_config.vllm_server_port,
        logging_steps=grpo_config.logging_steps,
        logging_first_step=grpo_config.logging_first_step,
        log_completions=grpo_config.log_completions,
        use_liger_kernel=grpo_config.use_liger_kernel,
    )
    grpo_trainer = CustomGRPOTrainer(
        model=base_model,
        reward_funcs=reward_func,
        train_dataset=dataset,
        peft_config=lora_config,
        optimizers=prepared_optimizer,
        args=grpo_config,
    )

    print('\n\n')
    print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
    print("GRPO Training Started...")
    print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
    print('\n\n')

    grpo_trainer.train()


if __name__ == "__main__":
    main()
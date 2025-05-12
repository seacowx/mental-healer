from pydantic import BaseModel


class GRPOTrainerArgs(BaseModel):
    bf16: bool
    use_liger_kernel: bool
    output_dir: str

    # sampling
    num_generations: int

    # training
    do_train: bool
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: int

    # optimizer
    adam_beta1: float
    adam_beta2: float
    weight_decay: float
    warmup_steps: int
    learning_rate: float

    # vllm settings
    use_vllm: bool
    trl_vllm_server_port: int
    trl_vllm_server_device: str

    # logging
    logging_steps: int
    logging_first_step: bool
    log_completions: bool

    # device allocation
    trl_vllm_server_device: str
    sentiment_reward_device: str

    # model path
    model_path: str
    base_agent_path: str

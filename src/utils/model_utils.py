
import torch
from utils.vllm_inference_utils import vLLMOffline


def load_offline_vllm_base_model(
    base_model_path: str,
    coping_chat_template_path: str = '',
    sentiment_reward_device: torch.device | None = None,
) -> vLLMOffline:

    extra_kwargs = {}
    if sentiment_reward_device:
        extra_kwargs['model_device'] = sentiment_reward_device
        extra_kwargs['tensor_parallel_size'] = 1
    else:
        extra_kwargs['tensor_parallel_size'] = torch.cuda.device_count()

    # initialize the llm
    offline_vllm_model = vLLMOffline(
        model_path=base_model_path,
        max_model_len=2048,
        enable_lora=True,
        max_lora_rank=64,
        gpu_memory_utilization=0.7,
        coping_chat_template_path=coping_chat_template_path,
        **extra_kwargs,
    )

    return offline_vllm_model


def load_all_models(
    base_model_path: str,
    coping_chat_template_path: str = '',
    sentiment_reward_device: torch.device | None = None,
): 

    base_offline_vllm_model = load_offline_vllm_base_model(
        base_model_path=base_model_path,
        sentiment_reward_device=sentiment_reward_device,
        coping_chat_template_path=coping_chat_template_path,
    )

    return base_offline_vllm_model

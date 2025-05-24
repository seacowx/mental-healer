
import torch
from utils.vllm_inference_utils import vLLMOffline


def load_offline_vllm_base_model(
    base_model_path: str,
    coping_chat_template_path: str = '',
    base_model_device: str | None = None,
    gpu_memory_utilization: float = 0.7,
) -> vLLMOffline:

    # initialize the llm
    offline_vllm_model = vLLMOffline(
        model_path=base_model_path,
        max_model_len=2048,
        enable_lora=True,
        max_lora_rank=64,
        gpu_memory_utilization=gpu_memory_utilization,
        coping_chat_template_path=coping_chat_template_path,
        model_device=base_model_device,
    )

    return offline_vllm_model


def load_all_models(
    base_model_path: str,
    coping_chat_template_path: str = '',
    base_model_device: str | None = None,
): 

    base_offline_vllm_model = load_offline_vllm_base_model(
        base_model_path=base_model_path,
        coping_chat_template_path=coping_chat_template_path,
        base_model_device=base_model_device,
    )

    return base_offline_vllm_model

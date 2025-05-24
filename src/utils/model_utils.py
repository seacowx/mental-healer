import os
import ray
import torch
import socket
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

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

    # Initialize Ray
    ray.init()
    num_models = 1
    pg = placement_group(
        name="llm_pg",
        bundles=[{"GPU": 1, "CPU": 1} for _ in range(num_models)],
        strategy="STRICT_PACK"  # or "PACK" or "SPREAD" depending on your needs
    )
    ray.get(pg.ready())

    @ray.remote(num_gpus=1, num_cpus=1)
    class LLMActor:
        def __init__(self, base_model_path):
            # Get the GPU IDs assigned to this actor by Ray
            gpu_ids = ray.get_gpu_ids()
            # Set CUDA_VISIBLE_DEVICES to limit the GPUs visible to this process
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(int(gpu_id)) for gpu_id in gpu_ids)
            # Set the default CUDA device
            torch.cuda.set_device(0)  # Since only one GPU is visible, it's cuda:0
            # Initialize the LLM model
            self.offline_vllm_model = load_offline_vllm_base_model(
                base_model_path=base_model_path,
                coping_chat_template_path=coping_chat_template_path,
                base_model_device=base_model_device,
            )

    # Create actors
    actors = []
    for i in range(num_models):
        # Assign the actor to a specific bundle in the placement group
        actor = LLMActor.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=i
            )
        ).remote(base_model_path)
        actors.append(actor)

        actor.offline_vllm_model.inference.remote(message_list=["Hello, how are you?"])

    raise SystemExit

    return base_offline_vllm_model

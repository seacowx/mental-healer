DEVICE_NUMBER=$(python3 -c "
import yaml
import re

def get_cuda_device_number(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    device_str = config.get('trl_vllm_server_device', 'cuda:0')
    match = re.search(r'cuda:(\d+)', device_str)
    return int(match.group(1)) if match else 0

print(get_cuda_device_number('./configs/grpo_config.yaml'))
")

MODEL_PATH=$(python3 -c "
import yaml

def get_model_path(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config.get('model_path', 'Qwen/Qwen3-4B')

print(get_model_path('./configs/grpo_config.yaml'))
")

TRL_VLLM_PORT=$(python3 -c "
import yaml

def get_port(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config.get('trl_vllm_server_port', 8880)

print(get_port('./configs/grpo_config.yaml'))
")

printf "\n\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n"
printf "Initializing TRL vLLM server with GPU CUDA:$DEVICE_NUMBER...\n"
printf "=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n\n"

CUDA_VISIBLE_DEVICES=$DEVICE_NUMBER trl vllm-serve \
    --model $MODEL_PATH \
    --port $TRL_VLLM_PORT \
    --tensor_parallel_size 1 \
    --gpu-memory-utilization 0.85
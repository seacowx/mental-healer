MODEL_PATH="Qwen/Qwen2-0.5B-Instruct"
TRL_VLLM_PORT=8880

# CUDA_VISIBLE_DEVICES=1 python debug.py \
#     --model_path "Qwen/Qwen2-0.5B-Instruct" \
#     --trl_vllm_port 8880

# STEP: initialize trl vllm server
CUDA_VISIBLE_DEVICES=1 python debug.py \
    --model_path $MODEL_PATH \
    --trl_vllm_port $TRL_VLLM_PORT

printf "\n\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n"
printf "Initializing vLLM server...\n"
printf "=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n\n"

# CUDA_VISIBLE_DEVICES=0 trl vllm-serve \
#     --model "Qwen/Qwen2-0.5B-Instruct" \
#     --port 8880 \
#     --tensor_parallel_size 1 \
#     --gpu-memory-utilization 0.75

CUDA_VISIBLE_DEVICES=0 trl vllm-serve \
    --model $MODEL_PATH \
    --port $TRL_VLLM_PORT \
    --tensor_parallel_size 1 \
    --gpu-memory-utilization 0.75

printf "\n\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n"
printf "vLLM server started. Starting training...\n"
printf "=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n\n"

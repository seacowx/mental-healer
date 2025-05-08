# MODEL_PATH="Qwen/Qwen2-0.5B-Instruct"
MODEL_PATH='Qwen/Qwen3-8B'
TRL_VLLM_PORT=8880


printf "\n\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n"
printf "Initializing vLLM server...\n"
printf "=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n\n"

CUDA_VISIBLE_DEVICES=0 trl vllm-serve \
    --model $MODEL_PATH \
    --port $TRL_VLLM_PORT \
    --tensor_parallel_size 1 \
    --gpu-memory-utilization 0.85

printf "\n\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n"
printf "vLLM server started. Now start the training script.\n"
printf "=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n\n"

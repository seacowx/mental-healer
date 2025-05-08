MODEL_PATH="Qwen/Qwen2-0.5B-Instruct"
TRL_VLLM_PORT=8880

# STEP: initialize trl vllm server
CUDA_VISIBLE_DEVICES=0 trl vllm-serve \
    --model $MODEL_PATH \
    --port $TRL_VLLM_PORT \
    --tensor_parallel_size 1 \
    --gpu-memory-utilization 0.75

printf "\n\n-----------------------------------------------------------------------"
printf "\n\nvLLM server started. Starting training..."
printf "-----------------------------------------------------------------------\n\n"

CUDA_VISIBLE_DEVICES=1 python debug.py \
    --model_path $MODEL_PATH \
    --trl_vllm_port $TRL_VLLM_PORT
# MODEL_PATH="Qwen/Qwen2-0.5B-Instruct"
MODEL_PATH='Qwen/Qwen3-4B'
TRL_VLLM_PORT=8880
TENSOR_PARALLEL_SIZE=$1

if [ -z "$TENSOR_PARALLEL_SIZE" ]; then
    TENSOR_PARALLEL_SIZE=1
fi

# make cuda visible devices to be 0 and 1 if TENSOR_PARALLEL_SIZE is 2
if [ $TENSOR_PARALLEL_SIZE -eq 2 ]; then
    CUDA_VISIBLE_DEVICES=0,1
else
    CUDA_VISIBLE_DEVICES=0
fi

printf "\n\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n"
printf "Initializing TRL vLLM server with $TENSOR_PARALLEL_SIZE GPUs (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)...\n"
printf "=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n\n"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES trl vllm-serve \
    --model $MODEL_PATH \
    --port $TRL_VLLM_PORT \
    --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
    --gpu-memory-utilization 0.85
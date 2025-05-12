MODEL_PATH='Qwen/Qwen3-8B'
TRL_VLLM_PORT=8000
TENSOR_PARALLEL_SIZE=$1

if [ -z "$TENSOR_PARALLEL_SIZE" ]; then
    TENSOR_PARALLEL_SIZE=1
fi

# make cuda visible devices to be 0 and 1 if TENSOR_PARALLEL_SIZE is 2
if [ $TENSOR_PARALLEL_SIZE -eq 2 ]; then
    CUDA_VISIBLE_DEVICES=2, 3
else
    CUDA_VISIBLE_DEVICES=1
fi

printf "\n\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n"
printf "Initializing Base vLLM server with $TENSOR_PARALLEL_SIZE GPUs (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)...\n"
printf "=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n\n"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES vllm serve \
    --model $MODEL_PATH \
    --served-model-name vllm-model \
    --task generate \
    --port $TRL_VLLM_PORT \
    --api-key anounymous123 \
    --max-model-len 4096 \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --gpu-memory-utilization 0.7 \
    --enforce-eager \
    --disable-log-requests \
    --lora-modules \
        {
            name: sentiment_reward, \
            path: /scratch/prj/charnu/ft_weights/mental-healer/reward-sentiment/qwen8/checkpoint-260, \
            base_model_name: vllm-model, \
        }

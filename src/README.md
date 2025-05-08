1. First initialize the vLLM server:

```bash
./init_vllm.sh 1
```
where `1` is the number of GPUs to use. The script will automatically allocate GPUs with smaller indices to vLLM. For example, if you have 4 GPUs, you can run:

```bash
./init_vllm.sh 2
```

and the script will allocate GPUs 0 and 1 to vLLM.

2. Then run the training script:

```bash
python run.py --base_model [BASE_MODEL] --training_config [TRAINING_CONFIG]
```

Arguments:
a. --base_model: the path to the base model.
b. --training_config: the path to the training configuration file.







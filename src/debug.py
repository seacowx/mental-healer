from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")

model.print_trainable_parameters()

import time
time.sleep(600)

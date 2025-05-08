from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
model.generate("Hello, how are you?")

import time
time.sleep(600)

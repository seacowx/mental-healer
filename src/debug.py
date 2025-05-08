from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

cur_input = "Hello, how are you?"
input_ids = tokenizer(cur_input, return_tensors="pt")

output = model.generate(**input_ids, max_length=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))

import time
time.sleep(600)

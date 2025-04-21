import yaml, json

import torch
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput
from vllm.lora.request import LoRARequest


class SentimentReward:


    def __init__(self) -> None:
        model_path_dict = yaml.safe_load(open('./configs/llm_configs.yaml'))
        model_path = model_path_dict['qwen7']['path']

        # initialize the llm
        reward_device = torch.device('cuda:0')
        self.llm = LLM(
            model=model_path, 
            max_model_len=2048,
            enable_lora=True,
            max_lora_rank=64,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,
            device=reward_device,
        )

        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=128,
            # stop=["</emotion>"],
        )

        self.adapter_dir = (
            '/scratch/prj/charnu/ft_weights/mental-healer/' 
            'reward-sentiment/qwen7/checkpoint-220'
        )


    def __parse_output(self, output: RequestOutput) -> str:

        out_str = ""
        try:
            out_str = output.outputs[0].text \
                .split('<emotion>')[1] \
                .split('</emotion>')[0].strip().lower() \
                .replace('"', '') \
                .replace("'", '') 
        except:
            pass

        return out_str


    def get_sentiment(self, input_msg_list: list) -> list:

        # keep track of the completed and corrupted outputs
        queue_list = list(range(len(input_msg_list)))

        out_list = [''] * len(input_msg_list)
        TOLERANCE = 5
        temperature = 0.0
        tol_counter = 0
        finished_idx_list = []
        while queue_list and tol_counter < TOLERANCE:

            outputs = self.llm.chat(
                messages=input_msg_list,
                sampling_params=self.sampling_params,
                lora_request=LoRARequest(f"sentiment", 1, self.adapter_dir),
                use_tqdm=True,
            )

            for output_idx, output in enumerate(outputs):
                parsed_output = self.__parse_output(output)

                if not parsed_output:
                    continue

                out_list[queue_list[output_idx]] = parsed_output
                finished_idx_list.append(output_idx)

            # update queue_list, remove finised idx
            queue_list = [queue_list[i] for i in range(len(queue_list)) if i not in finished_idx_list]
            input_msg_list = [input_msg_list[i] for i in range(len(input_msg_list)) if i not in finished_idx_list]
            tol_counter += 1

            # increment temperature
            self.sampling_params.temperature += 0.2

        # if there are remaining corrupted outputs, set them to be negative sentiment
        if queue_list:
            for idx in queue_list:
                out_list[idx] = 'negative'

        return out_list

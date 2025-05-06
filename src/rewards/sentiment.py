import gc
import yaml, json

import torch
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput
from vllm.lora.request import LoRARequest
from vllm.distributed.parallel_state import destroy_model_parallel


class SentimentReward:


    def __init__(
        self, 
        sentiment_reward_device: torch.device,
        llm_config_path: str = './configs/llm_configs.yaml', 
        reward_rule_path: str = './configs/sentiment_reward_rules.yaml',
    ) -> None:
        model_path_dict = yaml.safe_load(open(llm_config_path, 'r'))
        self.model_path = model_path_dict['qwen7']['path']

        self.reward_mapping = yaml.load(
            open(reward_rule_path, 'r'),
            Loader=yaml.FullLoader,
        )

        self.sentiment_reward_device = sentiment_reward_device

    
    def initialize_sentiment_reward_model(self):

        # initialize the llm
        self.llm = LLM(
            model=self.model_path, 
            max_model_len=2048,
            enable_lora=True,
            max_lora_rank=64,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,
            device=self.sentiment_reward_device,
        )

        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=128,
        )

        self.adapter_dir = (
            '/scratch/prj/charnu/ft_weights/mental-healer/' 
            'reward-sentiment/qwen7/checkpoint-220'
        )


    def terminate_sentiment_reward_model(self):
        destroy_model_parallel()
        del self.llm
        gc.collect()
        torch.cuda.empty_cache()
        torch.distributed.destroy_process_group()


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


    def get_sentiment(
        self, 
        input_msg_list: list, 
    ) -> list:

        # keep track of the completed and corrupted outputs
        remaining_indices = list(range(len(input_msg_list)))
        out_list = [''] * len(input_msg_list)
        TOLERANCE = 5
        tol_counter = 0

        while input_msg_list and tol_counter < TOLERANCE:

            outputs = self.llm.chat(
                messages=input_msg_list,
                sampling_params=self.sampling_params,
                lora_request=LoRARequest(f"sentiment", 1, self.adapter_dir),
                use_tqdm=True,
            )

            new_input_msg_list = []
            new_remaining_indices = []

            for i, output in enumerate(outputs):
                parsed_output = self.__parse_output(output)
                if parsed_output:
                    # Store the parsed result in the original index
                    out_list[remaining_indices[i]] = parsed_output
                else:
                    # If parsing failed, queue this message for the next iteration.
                    new_input_msg_list.append(input_msg_list[i])
                    new_remaining_indices.append(remaining_indices[i])

            input_msg_list = new_input_msg_list
            remaining_indices = new_remaining_indices

            tol_counter += 1
            self.sampling_params.temperature += 0.2

        for idx in remaining_indices:
            out_list[idx] = 'negative'

        # reset temperature
        self.sampling_params.temperature = 0.0

        return out_list


    def compute_sentiment_reward(
        self,
        new_sentiment_list: list, 
        previous_sentiment_list: list,
    ) -> list:

        reward_list = []
        for prev_sentiment, new_sentiment in zip(previous_sentiment_list, new_sentiment_list):
            reward_list.append(
                self.reward_mapping[prev_sentiment][new_sentiment]
            )
        reward_list = reward_list

        return reward_list
        


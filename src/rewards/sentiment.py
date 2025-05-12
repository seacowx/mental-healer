import gc
import yaml, json

import torch
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput
from vllm.lora.request import LoRARequest

from utils.vllm_inference_utils import vLLMServer


class SentimentReward:


    def __init__(
        self, 
        model_path: str,
        sentiment_reward_device: torch.device,
        reward_rule_path: str = './configs/sentiment_reward_rules.yaml',
        sentiment_mapping_path: str = './configs/emotion_to_sentiment.yaml',
    ) -> None:

        self.reward_mapping = yaml.safe_load(open(reward_rule_path, 'r'))
        self.sentiment_mapping = yaml.safe_load(open(sentiment_mapping_path, 'r'))
        # base vLLM server is shared between Patient Agent and Reward Model
        # Reward model will activate the corresponding LoRA adapter
        self.model_path = model_path
        self.sentiment_reward_device = sentiment_reward_device

    
    def initialize_sentiment_reward_model(self) -> LLM:

        extra_kwargs = {}
        if self.sentiment_reward_device:
            extra_kwargs['device'] = self.sentiment_reward_device
            extra_kwargs['tensor_parallel_size'] = 1
        else:
            extra_kwargs['tensor_parallel_size'] = torch.cuda.device_count()

        # initialize the llm
        self.llm = LLM(
            model=self.model_path, 
            max_model_len=2048,
            enable_lora=True,
            max_lora_rank=64,
            gpu_memory_utilization=0.7,
            **extra_kwargs,
        )

        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=128,
        )

        self.adapter_dir = (
            '/scratch/prj/charnu/ft_weights/mental-healer/reward-sentiment/qwen8/checkpoint-260'
        )

        return self.llm


    def __parse_output(self, output: RequestOutput) -> str:

        out_str = ""
        try:
            out_str = output.outputs[0].text \
                .split('<sentiment>')[1] \
                .split('</sentiment>')[0].strip().lower() \
                .replace('"', '') \
                .replace("'", '') 
        except:
            return ''

        out_str = self.sentiment_mapping.get(out_str, '')

        return out_str


    def get_sentiment(
        self, 
        input_msg_list: list, 
    ) -> list:

        # keep track of the completed and corrupted outputs

        outputs = self.llm.chat(
            messages=input_msg_list,
            sampling_params=self.sampling_params,
            lora_request=LoRARequest(f"sentiment", 1, self.adapter_dir),
            use_tqdm=True,
        )

        out_list = [''] * len(input_msg_list)
        for i, output in enumerate(outputs):
            parsed_output = self.__parse_output(output)
            if parsed_output:
                # Store the parsed result in the original index
                out_list[i] = parsed_output
            else:
                out_list[i] = 'positive'

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
        


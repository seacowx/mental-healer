import gc
import yaml, json

from vllm import SamplingParams
from vllm.outputs import RequestOutput
from vllm.lora.request import LoRARequest
from utils.vllm_inference_utils import vLLMOffline


class SentimentReward:


    def __init__(
        self, 
        base_vllm_model: vLLMOffline,
        reward_rule_path: str = './configs/sentiment_reward_rules.yaml',
        sentiment_mapping_path: str = './configs/emotion_to_sentiment.yaml',
        temperature: float = 0.,
        max_tokens: int = 128,
    ) -> None:

        self.reward_mapping = yaml.safe_load(open(reward_rule_path, 'r'))
        self.sentiment_mapping = yaml.safe_load(open(sentiment_mapping_path, 'r'))
        # base vLLM server is shared between Patient Agent and Reward Model
        # Reward model will activate the corresponding LoRA adapter

        self.temperature = temperature
        self.max_tokens = max_tokens

        self.adapter_dir = (
            '/scratch/prj/charnu/ft_weights/mental-healer/reward-sentiment/qwen8/checkpoint-260'
        )

        self.llm = base_vllm_model


    def __parse_output(self, output: str) -> str:

        out_str = ""
        try:
            out_str = output \
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

        outputs = self.llm.inference(
            message_list=input_msg_list,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
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
        


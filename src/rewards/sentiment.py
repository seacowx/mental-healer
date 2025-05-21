import gc
import yaml, json
from jinja2 import Template

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
        sentiment_prompt_path: str = './prompts/sentiment.yaml',
        temperature: float = 0.,
        max_tokens: int = 128,
        num_turns: int = 3,
        decay_factor: float = 0.5,
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
        self.sentiment_prompt = Template(yaml.safe_load(open(sentiment_prompt_path, 'r'))['input'])

        self.llm = base_vllm_model
        self.efficiency_reward_sequence = self._allocate_efficiency_reward(
            num_turns=num_turns, 
            decay_factor=decay_factor
        )

        print(self.efficiency_reward_sequence)
        raise SystemExit


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

    
    def _allocate_efficiency_reward(
            self,
            num_turns: int,
            decay_factor: float = 0.5,
    ) -> list:
        sequence = [1 * (decay_factor ** i) for i in range(num_turns)]
        return sequence

    
    def _compute_efficiency_reward(
        self,
        sentiment_list: list,
    ) -> list:
        ...


    def get_sentiment(
        self, 
        situation_desc_list: list,
        thought_list: list,
        show_vllm_tqdm_bar: bool = False,
    ) -> list:

        self.num_sample = len(situation_desc_list)
        self.num_thought = len(thought_list[0])

        input_msg_list = []
        input_msg_idx_list = []
        filled_idx_list, empty_idx_list = [], []
        for situation_idx, situation_desc in enumerate(situation_desc_list):
            cur_thought_list = thought_list[situation_idx]

            for thought_idx, thought in enumerate(cur_thought_list):
                if not thought:
                    empty_idx_list.append((situation_idx, thought_idx))
                else:
                    cur_input_msg = self.sentiment_prompt.render(
                        situation=situation_desc,
                        thought=thought,
                    )
                    cur_input_msg = [
                        {'role': 'user', 'content': cur_input_msg}
                    ]
                    input_msg_list.append(cur_input_msg)
                    filled_idx_list.append((situation_idx, thought_idx))

                input_msg_idx_list.append((situation_idx, thought_idx))

        outputs = self.llm.inference(
            message_list=input_msg_list,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            lora_request=LoRARequest(f"sentiment", 1, self.adapter_dir),
            show_tqdm_bar=show_vllm_tqdm_bar,
        )

        output_list = [[''] * self.num_thought] * self.num_sample
        output_idx = 0
        for output_msg_idx in input_msg_idx_list:
            if output_msg_idx in filled_idx_list:
                sentiment_output = self.__parse_output(outputs[output_idx])
                output_idx += 1

                if sentiment_output not in ['positive', 'negative']:
                    sentiment_output = self.sentiment_mapping[sentiment_output]
                sample_idx, thought_idx = output_msg_idx                

                output_list[sample_idx][thought_idx] = sentiment_output

        return output_list


    def compute_sentiment_reward(
        self,
        sentiment_list: list,
    ) -> list:

        reward_list = []

        print(sentiment_list)
        raise SystemExit


        return reward_list
        


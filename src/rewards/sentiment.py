import gc
import yaml, json
from asyncio import Semaphore
from tqdm.asyncio import tqdm as atqdm

import torch
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput
from vllm.lora.request import LoRARequest

from utils.vllm_inference_utils import OpenAIAsyncInference


class SentimentReward:


    def __init__(
        self, 
        base_vllm_client: OpenAIAsyncInference,
        reward_rule_path: str = './configs/sentiment_reward_rules.yaml',
    ) -> None:

        self.reward_mapping = yaml.load(
            open(reward_rule_path, 'r'),
            Loader=yaml.FullLoader,
        )
        # base vLLM server is shared between Patient Agent and Reward Model
        # Reward model will activate the corresponding LoRA adapter
        self.base_vllm_client = base_vllm_client

        self.sentiment_reward_device = sentiment_reward_device


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


    async def get_sentiment(
        self, 
        input_msg_list: list, 
    ) -> list:

        # keep track of the completed and corrupted outputs
        remaining_indices = list(range(len(input_msg_list)))
        out_list = [''] * len(input_msg_list)
        TOLERANCE = 5
        tol_counter = 0

        semaphore = Semaphore(50)

        while input_msg_list and tol_counter < TOLERANCE:

            outputs = [
                self.base_vllm_client.process_with_semaphore(
                    semaphore=semaphore,
                    model='vllm-model',
                    message=msg,
                )
                for msg in input_msg_list
            ]

            outputs = await asyncio.gather(*outputs)

            print(outputs[0])
            raise SystemExit

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
        


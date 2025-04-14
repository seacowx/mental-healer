import json
import os, sys
from copy import deepcopy

import torch

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(root_dir)

from src.utils.llm_inference import vLLMServer, OpenAIAsyncInference


class AppraisalPredictor:

    def __init__(
        self, 
        data: dict,
        prompt_template: dict,
        model_name: str,
        model_path_dict: dict,
    ) -> None:

        self.data = data
        self.prompt_template = prompt_template
        self.model_name = model_name
        self.model_path_dict = model_path_dict
        
        # laod model weight path and model quantization config
        model_config = self.model_path_dict[self.model_name]
        self.model_path = model_config['path']
        self.quantization = model_config['quantization']


    def predict(self):

        # initialize the vllm server
        vllm_server = vLLMServer(
            model_path=self.model_path,
            world_size=torch.cuda.device_count(),
            quantization=self.quantization,
        )

        vllm_server, openai_async_client = vllm_server.start_vllm_server()
        raise SystemExit()

        appraisal_desc_msg_list = []
        for entry_idx, (id, entry) in enumerate(self.data.items()):
            cur_situation = entry['context'].strip()
            cur_appraisal_prediction_prompt = deepcopy(self.prompt_template)
            cur_appraisal_prediction_prompt[1]['content'] = cur_appraisal_prediction_prompt[1]['content'].replace(
                '{{scenario}}', cur_situation
            )

            appraisal_desc_msg_list.append(cur_appraisal_prediction_prompt)

            print(cur_appraisal_prediction_prompt)
            raise SystemExit()

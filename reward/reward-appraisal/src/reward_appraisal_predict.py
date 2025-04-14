import json
import asyncio
import os, sys
import numpy as np
from copy import deepcopy
from tqdm.asyncio import tqdm as atqdm

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


    async def predict(self, appraisal_mtx: np.ndarray) -> np.ndarray:

        # initialize the vllm server
        vllm_server = vLLMServer(
            model_path=self.model_path,
            world_size=torch.cuda.device_count(),
            quantization=self.quantization,
        )

        openai_async_client = vllm_server.start_vllm_server()

        try:
            appraisal_desc_msg_list = []
            for entry_idx, (id, entry) in enumerate(self.data.items()):
                cur_situation = entry['context'].strip()
                cur_appraisal_prediction_prompt = deepcopy(self.prompt_template)
                cur_appraisal_prediction_prompt[1]['content'] = cur_appraisal_prediction_prompt[1]['content'].replace(
                    '{{scenario}}', cur_situation
                )

                appraisal_desc_msg_list.append(cur_appraisal_prediction_prompt)

            semaphore = asyncio.Semaphore(50)
            appraisal_pred_response_list = [openai_async_client.process_with_semaphore(
                semaphore=semaphore,
                model=self.model_name,
                message=msg,
                return_json=False,
            ) for msg in appraisal_desc_msg_list]

            appraisal_desc_list = await atqdm.gather(*appraisal_pred_response_list)

            # parse the appraisal prediction response
            appraisal_desc_list = [
                ele.split('<ratings>')[1].strip() if '<ratings>' in ele else ele
                for ele in appraisal_desc_list
            ]
            appraisal_desc_list = [
                ele.split('</ratings>')[0].strip() if '</ratings>' in ele else ele
                for ele in appraisal_desc_list
            ]

            valid_idx = 0
            for i, entry in enumerate(appraisal_desc_list):
                cur_predicted_appraisal_dims = entry.split('\n')
                cur_predicted_appraisal_dims = [
                    ele.replace('-', '').split(':') for ele in cur_predicted_appraisal_dims if ele.strip()
                ]

                cur_predicted_appraisal_dims_list = []
                for entry in cur_predicted_appraisal_dims:
                    if len(entry) == 2:
                        try:
                            cur_predicted_appraisal_dims_list.append(int(entry[1].strip()))
                        except:
                            continue

                if len(cur_predicted_appraisal_dims_list) == 21:
                    appraisal_mtx[valid_idx] = cur_predicted_appraisal_dims_list
                    valid_idx += 1

            print(appraisal_mtx.shape)
            raise SystemExit()

        finally:
            vllm_server.kill_server() 


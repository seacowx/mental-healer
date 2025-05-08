import os
import time
import json
import signal
import subprocess
from abc import ABCMeta, abstractmethod

import torch
from vllm import LLM, SamplingParams

import backoff
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai import (
    RateLimitError,
    APIError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
    BadRequestError,
)


class LLMBaseModel(metaclass=ABCMeta):


    def init_model(
        self,
        **kwargs,
    ) -> None:
        self.client = OpenAI(
            **kwargs,
        )


    def init_async_model(
        self,
        **kwargs,
    ):
        self.async_client = AsyncOpenAI(
            **kwargs,
        )


    @abstractmethod
    def inference(
        self, 
        model: str,
        message: list,
        temperature: float = 0.0, 
        max_tokens: int = 1024, 
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        return_json: bool = False,
        stream: bool = False,
        json_schema: dict = {},
    ) -> str | None | ChatCompletion:
        ...


class OpenAIAsyncInference(LLMBaseModel):
    """
    OpenAIAsyncInference class for async inference with OpenAI APIError
    Compatible with both official OpenAI API and vLLM server
    """

    def __init__(
        self,
        **kwargs,
    ) -> None:
        self.init_async_model(
            **kwargs,
        )


    async def process_with_semaphore(
        self, 
        semaphore, 
        model: str,
        message: list,
        temperature: float = 0.0, 
        top_p: float = 0.95,
        max_tokens: int = 1024, 
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        return_json: bool = False,
        json_schema = {},
    ):
        async with semaphore:
            return await self.inference(
                model=model,
                message=message,
                temperature=temperature, 
                top_p=top_p,
                max_tokens=max_tokens, 
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                return_json=return_json,
                json_schema=json_schema,
            )


    @backoff.on_exception(
        backoff.expo,
        (RateLimitError, APIError, APITimeoutError, APIConnectionError, InternalServerError),
        max_tries=5,
        max_time=70,
    )
    async def inference(
        self, 
        model: str,
        message: list,
        temperature: float = 1.0, 
        max_tokens: int = 1024, 
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        top_p: float = 0.95,
        return_json: bool = False,
        json_schema = None,
    ) -> str | ChatCompletion | None:

        # add generation prompt for non-GPT models
        extra_body = {}
        if 'gpt' not in model.lower():
            extra_body['add_generation_prompt'] = True

        if json_schema:
            extra_body['guided_json'] = json_schema
            extra_body['guided_decoding_backend'] = 'outlines'

        response = await self.async_client.chat.completions.create(
            model='vllm-model',
            messages=message,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            extra_body=extra_body,
        )

        if return_json:
            return response
        elif 'emollama' in model.lower():
            return response.choices[0].text
        else:
            return response.choices[0].message.content

    
class trlServer:

    def __init__(
        self,
        model_path: str,
        available_cuda_list: list[int] = [1],
    ) -> None:

        self.model_path = model_path
        self.available_cuda_list = available_cuda_list

    
    def start_trl_vllm_server(
        self,
        trl_vllm_port: int = 8880,
    ):
        env = os.environ.copy()
        server_command = [        
            'trl',        
            'vllm-serve',        
            '--model', self.model_path,    
            '--port', str(trl_vllm_port),
            '--tensor_parallel_size', str(len(self.available_cuda_list)),
            '--gpu-memory-utilization', '0.90',
        ]    

        env['CUDA_VISIBLE_DEVICES'] = ','.join([str(ele) for ele in self.available_cuda_list])

        # check if the server is running
        self.server = subprocess.Popen(
            args=server_command,
            env=env,
        )    
        server_running = False    
        number_of_attempts = 0    

        # large models such as Llama3.3-70B takes time to load
        # wait for 20 minutes for the server to start
        while (not server_running) and number_of_attempts < 120:        
            time.sleep(10)        
            result = subprocess.run(            
                [
                    "curl", f"http://localhost:{trl_vllm_port}/health/",
                ],             
                capture_output=True,             
                text=True,
            )        

            print('-------------------------------------------------')
            print(result.stdout)
            print('-------------------------------------------------')

            if 'vllm-model' in result.stdout or 'gpt-3.5-turbo' in result.stdout:            
                server_running = True        

            number_of_attempts += 1    

        if not server_running:        
            self.kill_server()        
            raise Exception("vllm-server failed to start")    


class vLLMServer:

    def __init__(
        self,
        model_path: str, 
        world_size: int, 
        quantization: bool,
        vllm_api_port: int = 8000,
    ) -> None:

        self.model_path = model_path
        self.world_size = world_size
        self.quantization = quantization
        self.vllm_api_port = vllm_api_port


    def start_vllm_server(
        self,
        device_list: list[int] = [],
    ) -> OpenAIAsyncInference:    

        model_config = json.load(
            open(os.path.join(self.model_path, 'config.json'), 'r')
        )
        
        # default max model len to 8192
        # check if the model has max_position_embeddings, if so, set max_model_len to the minimum of the two
        max_model_len = 8192
        if (max_position_embedding := model_config.get('max_position_embeddings', '')):
            max_model_len = min(max_position_embedding, 8192)

        # enable reasoning for Qwen3 models
        reasoning_params = []
        if 'qwen3' in self.model_path:
            reasoning_params = [
                '--reasoning-parser', 'qwen3',
                '--enable-reasoning',
            ]

        env = os.environ.copy()
        server_command = [        
            'vllm',        
            'serve',        
            self.model_path,    
            '--served-model-name', 'vllm-model',
            '--task', 'generate',
            '--port', str(self.vllm_api_port),
            '--api-key', 'anounymous123',
            '--max-model-len', str(max_model_len),
            '--tensor-parallel-size', str(self.world_size),
            '--gpu-memory-utilization', '0.95',
            '--enforce-eager', 
        ]    

        # add reasoning params if model is Qwen3
        if reasoning_params:
            server_command.extend(reasoning_params)

        if self.quantization:
            quantization_command = [
                '--load-format', 'bitsandbytes',
                '--quantization', 'bitsandbytes',
            ]
            server_command.extend(quantization_command)

        # add api key to environemnt variable 
        env['VLLM_API_KEY'] = 'anounymous123'
        if device_list:
            env['CUDA_VISIBLE_DEVICES'] = ','.join([str(ele) for ele in device_list])

        # check if the server is running
        self.server = subprocess.Popen(
            args=server_command,
            env=env,
        )    
        server_running = False    
        number_of_attempts = 0    
        # large models such as Llama3.3-70B takes time to load
        # wait for 20 minutes for the server to start
        while (not server_running) and number_of_attempts < 120:        
            time.sleep(10)        
            result = subprocess.run(            
                [
                    "curl", f"http://localhost:{self.vllm_api_port}/v1/models",
                    "--header", "Authorization: Bearer anounymous123",
                ],             
                capture_output=True,             
                text=True,
            )        

            if 'vllm-model' in result.stdout or 'gpt-3.5-turbo' in result.stdout:            
                server_running = True        

            number_of_attempts += 1    

        if not server_running:        
            self.kill_server()        
            raise Exception("vllm-server failed to start")    

        openai_server = OpenAIAsyncInference(
            base_url=f'http://localhost:{self.vllm_api_port}/v1/',
            api_key='anounymous123',
        )

        return openai_server


    def kill_server(self):    
        self.server.send_signal(signal.SIGINT)    
        self.server.wait()    
        time.sleep(10)


class vLLMOffline:


    def __init__(
        self,
        model_path: str, 
        quantization: str = '',
        max_model_len: int = 2048,
        patient_device: torch.device = None,
        gpu_memory_utilization=0.8,
    ) -> None:

        self.model_path = model_path
        self.quantization = quantization

        vllm_config = {}
        vllm_config['max_model_len'] = max_model_len
        vllm_config['tensor_parallel_size'] = 1 if patient_device else torch.cuda.device_count()
        vllm_config['device'] = patient_device if patient_device else 'auto'
        vllm_config['gpu_memory_utilization'] = gpu_memory_utilization
        vllm_config['quantization'] = quantization if quantization else None

        self.vllm_model = LLM(
            model=self.model_path,
            **vllm_config,
        )


    def inference(
        self, 
        message_list: list, 
        **kwargs
    ) -> list:
        """
        Inference with vLLM model
        """
        
        # NOTE: Default parameters are set according to the recommendations from the Qwen3 model hub
        # https://huggingface.co/Qwen/Qwen3-32B
        sampling_params = SamplingParams(
            temperature=kwargs.get('temperature', 0.6),
            max_tokens=kwargs.get('max_tokens', 8192),
            top_p=kwargs.get('top_p', 0.95),
            top_k=kwargs.get('top_k', 20),
            frequency_penalty=kwargs.get('frequency_penalty', 0.0),
            presence_penalty=kwargs.get('presence_penalty', 1.0),
            stop=kwargs.get('stop', None),
        )

        enable_thinking = kwargs.get('enable_thinking', False)

        response = self.vllm_model.chat(
            messages=message_list, 
            sampling_params=sampling_params,
            chat_template_kwargs={
                "enable_thinking": enable_thinking,
            },
            use_tqdm=True,
        )

        response = [ele.outputs[0].text for ele in response]

        return response
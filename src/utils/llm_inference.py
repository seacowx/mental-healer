import os
import time
import json
import signal
import subprocess
from abc import ABCMeta, abstractmethod

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
        top_p: float = 0.95,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        do_sample: bool = False,
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
        max_tokens: int = 1024, 
        top_p: float = 0.95,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        do_sample: bool = False,
        return_json: bool = False,
        stream: bool = False,
        json_schema = {},
    ):
        async with semaphore:
            return await self.inference(
                model=model,
                message=message,
                temperature=temperature, 
                max_tokens=max_tokens, 
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                do_sample=do_sample,
                return_json=return_json,
                stream=stream,
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
        top_p: float = 0.95,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        do_sample: bool = False,
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
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            extra_body=extra_body,
            do_sample=do_sample,
        )

        if return_json:
            return response
        elif 'emollama' in model.lower():
            return response.choices[0].text
        else:
            return response.choices[0].message.content


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
    ) -> tuple:    

        model_config = json.load(
            open(os.path.join(self.model_path, 'config.json'), 'r')
        )
        
        # default max model len to 8192
        # check if the model has max_position_embeddings, if so, set max_model_len to the minimum of the two
        max_model_len = 8192
        if (max_position_embedding := model_config.get('max_position_embeddings', '')):
            max_model_len = min(max_position_embedding, 8192)

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
            '--pipeline-parallel-size', str(self.world_size),
            '--gpu-memory-utilization', '0.95',
        ]    

        if self.quantization:
            quantization_command = [
                '--load-format', 'bitsandbytes',
                '--quantization', 'bitsandbytes',
                '--enforce-eager', 
            ]
            server_command.extend(quantization_command)

        # add api key to environemnt variable 
        os.environ['VLLM_API_KEY'] = 'anounymous123'

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

        return self.server, openai_server


    def kill_server(self):    
        self.server.send_signal(signal.SIGINT)    
        self.server.wait()    
        time.sleep(10)


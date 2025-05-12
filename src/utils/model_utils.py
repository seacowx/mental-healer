import yaml
from functools import wraps
from utils.vllm_inference_utils import vLLMServer


class ServerContainer:
    def __init__(self):
        self.servers: list[vLLMServer] = []

def ensure_graceful_exit(server_container: ServerContainer):
    """
    Decorator that ensures vllm servers are properly killed when the decorated function exits.
    
    Args:
        server_container: A container object that holds the list of vllm server instances
    """
    def decorator(func):
        @wraps(func)
        def wrapper_func(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            finally:
                for server in server_container.servers:
                    server.kill_server()
        return wrapper_func
    return decorator


def initialize_models_and_agents(
    patient_base_model: str,
):

    model_path_dict = yaml.safe_load(open('./configs/llm_configs.yaml', 'r'))
    agent_vllm_config = yaml.safe_load(open('./configs/agent_vllm_config.yaml', 'r'))
    patient_base_model_path = model_path_dict[patient_base_model]['path']

    print(
        '\n\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n'
        'Initializing base vLLM server for Patient Agent and Reward Model\n'
        '=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n\n'
    )    

    agent_vllm_server = vLLMServer(
        model_path=patient_base_model_path,
        **agent_vllm_config,
    )
    agent_vllm_server.start_vllm_server()

    return agent_vllm_server
    
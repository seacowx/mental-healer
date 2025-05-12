import yaml
from utils.vllm_inference_utils import vLLMServer


def initialize_models_and_agents(
    patient_base_model: str,
):

    model_path_dict = yaml.safe_load(open('./configs/llm_configs.yaml', 'r'))
    base_vllm_config = yaml.safe_load(open('./configs/base_vllm_config.yaml', 'r'))
    patient_base_model_path = model_path_dict[patient_base_model]['path']

    print(
        '\n\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n'
        'Initializing base vLLM server for Patient Agent and Reward Model\n'
        '=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n\n'
    )    

    base_vllm_server = vLLMServer(
        model_path=patient_base_model_path,
        **base_vllm_config,
    )
    base_vllm_server.start_vllm_server()
    
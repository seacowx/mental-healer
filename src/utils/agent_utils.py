import yaml
from agents.patient import Patient
from src.utils.vllm_inference_utils import vLLMServer


def initialize_patient_agent(
    patient_model: str,
) -> Patient:

    llm_path_dict = yaml.safe_load(open('./configs/llm_configs.yaml', 'r'))

    # STEP: initialize vllm server for patient agent, host the server on cuda:0
    patient_vllm_server = vLLMServer(
        model_path=llm_path_dict[patient_model]['path'],
        world_size=1,
        quantization=False,
    )
    patient_async_client = patient_vllm_server.start_vllm_server(
        device_list=[0],
    )

    patient_agent = Patient(
        openai_async_client=patient_async_client,
    )

    return patient_agent

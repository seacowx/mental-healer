from asyncio import Semaphore
from tqdm.asyncio import tqdm as atqdm

from openai import AsyncOpenAI

from utils.llm_inference_utils import vLLMServer
from rewards.therapist_reward import TherapistReward


# TODO: Check the implementation of the itrative thought generation function

def parse_thought_output(think_output_list: list) -> tuple[list, list]:
        
    parsed_output = []
    corrupted_idx_list = []
    for think_output_idx, think_output in enumerate(think_output_list):
        try:
            think_output = think_output.split('</think>')[-1].rsplit('<thought>')[1].split('</thought>')[0].strip()
            parsed_output.append(think_output)
        except Exception as e:
            parsed_output.append('')
            corrupted_idx_list.append(think_output_idx)

    return parsed_output, corrupted_idx_list


async def iterative_thought_generation(
    initial_thought_message_list: list,
    situation_list: list,
    therapist_reward: TherapistReward,
    vllm_client: vLLMServer,
    thought_device: list = [],
    TOLERANCE: int = 5,
):
    """
    Iteratively generate thoughts until the sentiment is negative or the number of iterations exceeds TOLERANCE.

    The generation process is iterative and follows these steps:
        1. Given a situation and persona profile, the agent generates an initial thought.
        2. The situation and the initial thought are passed to the sentiment reward model to get a sentiment result.
        3. The initial thought is valid if it results in negative sentiment. Otherwise, regenerate the thought.
        4. Steps 1-3 are repeated until all the initial thoughts result in negative sentiment.

    Args:
        initial_thought_message_list (list): List of messages for initial thought generation.
        situation_list (list): List of situations for sentiment analysis.
        therapist_reward (TherapistReward): TherapistReward object for sentiment analysis.
        queue_idx_list (list): List of indices to track which thoughts are still in the queue.
        vllm_client (vLLMServer): VLLM client for generating thoughts.
        thought_device (list): List of devices for thought generation.
        TOLERANCE (int): Maximum number of iterations to prevent infinite loops.

    Returns:
        valid_initial_thought_list (list): List of valid initial thoughts after sentiment analysis. Invalid thoughts are replaced with empty strings.
    """

    active_indices = list(range(len(initial_thought_message_list)))
    active_messages = initial_thought_message_list.copy()
    active_situations = situation_list.copy()
    valid_initial_thought_list = [''] * len(initial_thought_message_list)

    num_iterations = 0
    while active_messages and num_iterations < TOLERANCE:

        # initialize the async vllm server
        openai_async_server = vllm_client.start_vllm_server(device_list=thought_device)

        semaphore = Semaphore(50)

        # generate initial thoughts
        think_output_list = [
            openai_async_server.process_with_semaphore(
                semaphore=semaphore,
                model='vllm-model',
                message=active_message,
                temperature=0.6,
                max_tokens=4096,
                top_p=0.95,
                frequency_penalty=0.0,
                presence_penalty=1.0,
            )
            for active_message in active_messages[:100]
        ]

        think_output_list = await atqdm.gather(*think_output_list)

        # terminate the async vllm server
        vllm_client.kill_server()

        parsed_output, corrupted_idx_list = parse_thought_output(
            think_output_list=think_output_list,
        )

        # initialize the sentiment reward model
        therapist_reward.initialize_sentiment_reward_model()

        sentiment_msg_list = therapist_reward.make_sentiment_input_msg(
            situation_list=situation_list[:100],
            thoutght_list=parsed_output,
        )

        output_sentiment_list = therapist_reward.sentiment_reward.get_sentiment(
            input_msg_list=sentiment_msg_list,
        )

        # convert the sentiment label to "positive" for the corrupted output
        for idx in corrupted_idx_list:
            output_sentiment_list[idx] = 'positive'

        new_active_indices = []
        new_active_messages = []
        new_active_situations = []
        for i, sentiment in enumerate(output_sentiment_list):
            if sentiment == 'negative':
                valid_initial_thought_list[active_indices[i]] = parsed_output[i]
            else:
                # keep message for re-generation if sentiment is positive
                new_active_indices.append(active_indices[i])
                new_active_messages.append(active_messages[i])
                new_active_situations.append(active_situations[i])

        active_indices = new_active_indices
        active_messages = new_active_messages
        active_situations = new_active_situations

        # terminate the sentiment reward model
        therapist_reward.terminate_sentiment_reward_model()

        num_iterations += 1

    return valid_initial_thought_list

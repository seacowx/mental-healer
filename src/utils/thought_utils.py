from utils.llm_inference_utils import vLLMOffline
from modules.therapist_reward import TherapistReward


# TODO: Check the implementation of the itrative thought generation function

def iterative_thought_generation(
    initial_thought_message_list: list,
    situation_list: list,
    therapist_reward: TherapistReward,
    queue_idx_list: list,
    vllm_client: vLLMOffline,
    enable_thinking: bool = True,
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
        vllm_client (vLLMOffline): VLLM client for generating thoughts.
        TOLERANCE (int): Maximum number of iterations to prevent infinite loops.

    Returns:
        valid_initial_thought_list (list): List of valid initial thoughts after sentiment analysis. Invalid thoughts are replaced with empty strings.
    """

    print(enable_thinking)
    raise SystemExit()

    num_iterations = 0
    valid_initial_thought_list = [''] * len(initial_thought_message_list)
    while queue_idx_list and num_iterations < TOLERANCE:

        # generate initial thoughts
        think_output_list = vllm_client.inference(
            message_list=initial_thought_message_list,
            enable_thinking=enable_thinking,
        )
        
        parsed_output = []
        corrupted_idx_list = []
        for think_output_idx, think_output in enumerate(think_output_list):
            try:
                think_output = think_output.split('</think>')[-1].split('<thought>')[1].split('</thought>')[0]
                parsed_output.append(think_output)
            except Exception as e:
                parsed_output.append('')
                corrupted_idx_list.append(think_output_idx)

        sentiment_msg_list = therapist_reward.make_sentiment_input_msg(
            situation_list=situation_list,
            thoutght_list=parsed_output,
        )

        output_sentiment_list = therapist_reward.sentiment_reward.get_sentiment(
            input_msg_list=sentiment_msg_list,
        )

        # convert the sentiment label to "positive" for the corrupted output
        for idx in corrupted_idx_list:
            output_sentiment_list[idx] = 'positive'

        valid_queue_idx_list = [
            idx for idx, ele in enumerate(output_sentiment_list)
            if ele == 'negative'
        ]

        for round_idx, queue_idx in enumerate(valid_queue_idx_list):
            valid_initial_thought_list[queue_idx] = parsed_output[round_idx]

        # retrain the queue index if the sentiment is positive
        queue_idx_list = [
            idx for idx, ele in enumerate(output_sentiment_list)
            if ele == 'negative'
        ]
        initial_thought_message_list = [
            initial_thought_message_list[idx] for idx in queue_idx_list
        ]

        num_iterations += 1

    return valid_initial_thought_list

import json
import tqdm
from datasets import Dataset

from utils.persona_utils import sample_persona
from utils.persona_utils import retrieve_augmented_persona


def augment_situation_with_persona(
    data_path: str = '../data/situations/situations.json',
    n_personas=1,
) -> dict:
    """
    Prepare training data for the model.

    1. Load the situation data which contains the situation and candidate persona profiles.
    2. For each situation, sample a persona profile from the candidate persona profiles according to their density.

    Inputs:
        n_personas (int): The number of personas to sample for each situation. Default is 1. n_personas greater than 1 will duplicate the situation.

    Returns:
        prepared_data (dict): A dictionary where each key is a situation ID and the value is a dictionary containing the situation and the sampled persona profile.
    """

    data = json.load(open(data_path, 'r'))

    pbar = tqdm.tqdm(
        total=len(data) * n_personas,
        desc="Preparing training data",
    )

    prepared_data = {}
    for key, entry_dict in data.items():

        situation = entry_dict['situation']
        candidate_persona_info_list = entry_dict['candidate_persona_profile_list']

        sampled_persona_profile = sample_persona(
            candidate_persona_info_list=candidate_persona_info_list,
            n_personas=n_personas,
        )

        for i in range(1, n_personas+1):
            new_key = f"{key}||{i}"
            prepared_data[new_key] = {
                'situation': situation,
                'persona_profile': sampled_persona_profile[i-1],
            }

        pbar.update(n_personas)

    return prepared_data


def prepare_training_data(
        data_path: str,
) -> tuple[Dataset, Dataset, dict]:
    """
    Prepare training data for the model.

    1. Training data that initiate the GRPO training (conversation)
    2. Persona data for the Patient Agent

    Inputs:
        data_path (str): The path to the data file.

    Returns:
        tuple[Dataset, Dataset]: A tuple of two datasets. The first dataset is the training data that initiate the GRPO training (conversation). The second dataset is the persona data for the Patient Agent.
    """

    input_dict = json.load(open(data_path, 'r'))
    augmented_persona_profile_dict = retrieve_augmented_persona(situation_dict=input_dict)

    n_personas = int(data_path.split('/')[-1].split('.')[0][-1])

    pbar = tqdm.tqdm(
        total=len(input_dict) * n_personas,
        desc="Preparing training data...",
    )

    persona_data = {
        'id': [],
        'persona_profile': [],
    }
    conversation_data = {
        'id': [],
        'situation': [],
        'prompt': [],
    }
    augmented_input_dict = {}
    for key, val in input_dict.items():
        persona_data['id'].append(key)
        persona_data['persona_profile'].append(augmented_persona_profile_dict[key])

        situation_desc = val['situation']
        initial_thought = val['initial_thought']
        initial_thought_prompt = situation_desc + ' ' + initial_thought
        conversation_data['id'].append(key)
        # also need to add separated situation info for reward model
        # when new thought is generated, it will be concatenated with the situation info to compute sentiment reward
        conversation_data['situation'].append(situation_desc)
        conversation_data['prompt'].append(initial_thought_prompt)

        augmented_input_dict[key] = {
            'situation': situation_desc,
            'initial_thought': initial_thought,
            'initial_thought_prompt': initial_thought_prompt,
            'persona_profile': augmented_persona_profile_dict[key],
        }

        pbar.update(1)

    persona_data = Dataset.from_dict(persona_data)
    conversation_data = Dataset.from_dict(conversation_data)

    return conversation_data, persona_data, augmented_input_dict
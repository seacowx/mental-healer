import json
import tqdm
from utils.persona_utils import sample_persona


def prepare_training_data(
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

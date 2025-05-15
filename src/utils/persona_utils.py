import json
import numpy as np


def sample_persona(candidate_persona_info_list, n_personas=1) -> list:
    """
    Sample a persona profile from the candidate persona profiles according to their density
    """
    
    candidate_persona_profile_list = [
        ele['persona'] for ele in candidate_persona_info_list
    ]
    candidate_persona_prob_list = [
        ele['density'] for ele in candidate_persona_info_list
    ]
    
    # sample a persona profile according to the density
    sampled_persona_profile = np.random.choice(
        candidate_persona_profile_list, 
        p=candidate_persona_prob_list,
        size=n_personas,
        replace=False,
    )

    sampled_persona_profile =  [
        ele.replace('\n', ' ').strip() for ele in sampled_persona_profile
    ]

    sampled_persona_profile = [
        ele + '.' if not ele.endswith('.') else ele for ele in sampled_persona_profile
    ]

    return sampled_persona_profile


def locate_persona_idx(
    augmented_persona_dict: dict,
    persona_key_list: list,
    persona_desc: str,
) -> str:

    # remove the period from the persona description
    persona_desc = persona_desc.rstrip('.')

    for key in persona_key_list:

        augmented_persona_profile = augmented_persona_dict[key]
        cur_persona_desc = augmented_persona_profile['persona_hub'].rstrip('.')

        if persona_desc == cur_persona_desc:
            return key

    raise ValueError(f'Persona description\n"{persona_desc}"\nnot found in the augmented persona dictionary')


def retrieve_augmented_persona(
    situation_dict: dict, 
    matched_persona_path: str,
    persona_hub_path: str,
) -> dict:
    """
    Retrieve the augmented persona for the given situation
    """

    situation_to_persona_dict = json.load(open(matched_persona_path, 'r'))
    augmented_persona_dict = json.load(open(persona_hub_path, 'r'))

    out_augmented_persona_dict = {}
    for key, val in situation_dict.items():

        parsed_key = key.split('||')[0].strip()

        persona_idx_list = [ele['id'] for ele in situation_to_persona_dict[parsed_key]]
        augmented_persona_idx = locate_persona_idx(
            augmented_persona_dict=augmented_persona_dict,
            persona_key_list=persona_idx_list,
            persona_desc=val['persona_profile'],
        )

        augmented_persona_profile = augmented_persona_dict[augmented_persona_idx]

        out_augmented_persona_dict[key] = augmented_persona_profile

    return out_augmented_persona_dict


def verbalize_persona_profile(persona_profile_dict: dict) -> str:

    persona_profile_desc = (
        persona_profile_dict['persona_hub'] + '\n\nDetailed Persona Profile:\n'
        f'Name: {persona_profile_dict["name"]}\n'
        f'Gender: {persona_profile_dict["gender"]}\n'
        f'Occupation: {persona_profile_dict["occupation"]}\n'
        f'Education: {persona_profile_dict["education"]}\n'
        f'Personality: {persona_profile_dict["traits"]}\n'
    )

    return persona_profile_desc
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
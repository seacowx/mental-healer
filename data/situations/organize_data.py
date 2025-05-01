import re
import yaml, json
import pandas as pd


def main():

    situation_data =  json.load(
        open('../AugESC/augesc_content_filtered.json')
    )

    persona_data = json.load(
        open('../PersonaHub/persona.json')
    )

    matched_persona_data = json.load(
        open('../AugESC/augsec_matched_persona.json')
    )

    augmented_situation_data = {}
    for key, val in situation_data.items():

        matched_persona_info_list = matched_persona_data[key]
        matched_persona_id_list = [
            ele['id'] for ele in matched_persona_info_list
        ]
        matched_persona_prob_dist = [
            ele['density'] for ele in matched_persona_info_list
        ]

        matched_persona_profile_list = [
            {'persona': persona_data[persona_id], 'density': persona_prob}
            for persona_id, persona_prob in zip(matched_persona_id_list, matched_persona_prob_dist)
        ]

        augmented_situation_data[key] = {
            'situation': val,
            'candidate_persona_profile_list': matched_persona_profile_list,
        }

    with open('./situations.json', 'w') as f:
        json.dump(augmented_situation_data, f, indent=4)


if __name__ == "__main__":
    main()

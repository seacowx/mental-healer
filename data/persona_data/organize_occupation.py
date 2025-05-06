import json
import pandas as pd


def main():

    # Load the occupation data from the txt file
    occupation_df = pd.read_excel('./occupation_data/national_M2023_dl.xlsx', sheet_name='national_M2023_dl')
    all_occupations = occupation_df[occupation_df['O_GROUP'] == 'detailed'][['OCC_TITLE', 'TOT_EMP', 'OCC_CODE']]

    # remove entries where 'all other' is in the occupation title
    all_occupations = all_occupations[~all_occupations['OCC_TITLE'].str.contains('all other', case=False)]

    # organize the occupations into a dictionary, key is the occupation name, value is the normalized frequency
    total_employment = all_occupations['TOT_EMP'].sum()

    occupation_to_education_dict = {}
    for _, row in all_occupations.iterrows():  
        occupation = row['OCC_TITLE']
        frequency = row['TOT_EMP']
        occ_code = row['OCC_CODE']

        occupation_to_education_dict[occ_code] = {
            'occupation': occupation,
            'frequency': frequency / total_employment,
        }

    occupation_to_code = {
        val['occupation']: key
        for key, val in occupation_to_education_dict.items()
    }
    code_to_occupation = {
        key: val['occupation']
        for key, val in occupation_to_education_dict.items()
    }

    # STEP: match occupation with education level
    education_data = pd.read_excel('./occupation_data/education.xlsx', sheet_name='Table 5.3', header=1)

    # map the education levels from U.S. Bureau of Labor Statistics to the ones we use
    education_level_mapping = {
        'Less than high school diploma': 'high school',
        'High school diploma or equivalent': 'high school',
        'Some college, no degree': "high school",
        "Associate's degree": "bachelor's",
        "Bachelor's degree": "bachelor's",
        "Master's degree": "master's",
        'Doctoral or professional degree': 'PhD',
    }
    education_level_inversed_mapping = {
        'high school': [
            'Less than high school diploma', 
            'High school diploma or equivalent', 
            'Some college, no degree'
        ],
        "bachelor's": [
            "Associate's degree", 
            "Bachelor's degree"
        ],
        "master's": ["Master's degree"],
        'PhD': ['Doctoral or professional degree'],
    }

    for key, val in occupation_to_education_dict.items():

        education_distribution = education_data[education_data['2023 National Employment Matrix code'] == key]

        cur_education_dist = {}
        for col_name in education_distribution.columns:
            if col_name in education_level_mapping:
                education_name = education_level_mapping[col_name]
                cur_education_dist[education_name] = education_distribution[col_name].values[0]

        # normalize the education distribution
        total = sum(cur_education_dist.values())
        for edu_degree in cur_education_dist.keys():
            cur_education_dist[edu_degree] /= total

        occupation_to_education_dict[key]['education_distribution'] = cur_education_dist

    education_levels = ["high school", "bachelor's", "master's", "PhD"]
    education_to_occupation_dict = {level: [] for level in education_levels}

    for edu_level in education_to_occupation_dict.keys():
        for _, entry in education_data.iterrows():

            cur_edu_names = education_level_inversed_mapping[edu_level]
            cur_edu_prob = 0
            for cur_edu_name in cur_edu_names:
                cur_edu_prob += entry[cur_edu_name]

            cur_occ_code = entry['2023 National Employment Matrix code']
            cur_occ_name = code_to_occupation.get(cur_occ_code, None)

            # skip the entry if the occupation name is not found, these are non-detailed occupations
            if not cur_occ_name:
                continue

            education_to_occupation_dict[edu_level].append({
                'occupation': cur_occ_name,
                'probability': cur_edu_prob,
            })

    # for each education level, sort the occupations by probability
    for edu_level in education_to_occupation_dict.keys():
        education_to_occupation_dict[edu_level] = sorted(
            education_to_occupation_dict[edu_level],
            key=lambda x: x['probability'],
            reverse=True,
        )

    # save the top 100 occupations for each education level
    for edu_level in education_to_occupation_dict.keys():
        education_to_occupation_dict[edu_level] = education_to_occupation_dict[edu_level][:100]

    for edu_level, edu_info_dict_list in education_to_occupation_dict.items():
        total_prob = sum(
            [entry['probability'] for entry in education_to_occupation_dict[edu_level]]
        )
        for i in range(len(edu_info_dict_list)):
            education_to_occupation_dict[edu_level][i]['probability'] /= total_prob

    with open('./occupation_to_education.json', 'w') as f:
        json.dump(occupation_to_education_dict, f, indent=4)

    with open('./education_to_occupation.json', 'w') as f:
        json.dump(education_to_occupation_dict, f, indent=4)


if __name__ == "__main__":
    main()
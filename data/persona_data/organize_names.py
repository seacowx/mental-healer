import json
import pandas as pd
from glob import glob


def compute_name_frequency():
    """
    Compute and return the normalized frequency (relative frequency) of names 
    for each gender across all matching name files.
    
    Returns:
        pd.DataFrame: A DataFrame with columns ['gender', 'name', 'count', 'frequency'].
    """
    all_name_path_list = glob('./ssn_names/*.txt')

    male_name_dict, female_name_dict = {}, {}
    for name_path in all_name_path_list:
        name_df = pd.read_csv(name_path, header=None)
        name_df.columns = ['name', 'gender', 'count']

        # Group by gender and name, and sum the counts.
        grouped = (
            name_df.groupby(['gender', 'name'], as_index=False)
            .agg({'count': 'sum'})
        )

        # Compute the normalized frequency of each name within its gender group.
        grouped['frequency'] = grouped.groupby('gender')['count'].transform(lambda x: x / x.sum())

        male_names = grouped[grouped['gender'] == 'M']
        female_names = grouped[grouped['gender'] == 'F']

        # sort by frequenct
        male_names = male_names.sort_values(
            by='frequency', ascending=False, ignore_index=True
        ).head(100)
        female_names = female_names.sort_values(
            by='frequency', ascending=False, ignore_index=True
        ).head(100)

        # Add name and frequency to male_name_dict and female_name_dict. Name as key and frequency as value.
        for _, row in male_names.iterrows():

            name = row['name'].capitalize()
            frequency = row['frequency']

            if name in male_name_dict:
                male_name_dict[name] += frequency
            else:
                male_name_dict[name] = frequency

        for _, row in female_names.iterrows():
            name = row['name']
            frequency = row['frequency']

            if name in female_name_dict:
                female_name_dict[name] += frequency
            else:
                female_name_dict[name] = frequency

    # normalize the frequencies in male_name_dict and female_name_dict
    male_name_dict = {name: male_name_dict[name] for name in list(male_name_dict.keys())[:100]}
    male_frequency_sum = sum(male_name_dict.values())
    male_name_dict = {name: freq / male_frequency_sum for name, freq in male_name_dict.items()}
    male_name_dict = dict(sorted(male_name_dict.items(), key=lambda x: x[1], reverse=True))

    female_name_dict = {name: female_name_dict[name] for name in list(female_name_dict.keys())[:100]}
    female_frequency_sum = sum(female_name_dict.values())
    female_name_dict = {name: freq / female_frequency_sum for name, freq in female_name_dict.items()}
    female_name_dict = dict(sorted(female_name_dict.items(), key=lambda x: x[1], reverse=True))

    return male_name_dict, female_name_dict


def main():

    # Get the normalized name frequency
    male_name_dict, female_name_dict = compute_name_frequency()

    with open('./male_names.json', 'w') as f:
        json.dump(male_name_dict, f, indent=4)

    with open('./female_names.json', 'w') as f:
        json.dump(female_name_dict, f, indent=4)


if __name__ == "__main__":
    main()
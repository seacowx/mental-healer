import json
import random
import numpy as np
from jinja2 import Template
from thefuzz import fuzz, process


class BigFivePersonality:
    """
    A class to generate random big five personality traits.
    The traits are:
        - Neuroticism: Emotionally stable or unstable
        - Conscientiousness: Organized or careless
        - Extraversion: Extravert or introvert
        - Openness: Conventional or unconventional
        - Agreeableness: Agreeable or antagonistic
    """

    def __init__(self):

        self.neuroticism = ['emotionally stable', 'emotionally unstable']
        self.conscientiousness = ['organized', 'careless']
        self.extraversion = ['extroverted', 'introverted']
        self.openness = ['conservative', 'open-minded']
        self.agreeableness = ['agreeable', 'antagonistic']


    def __call__(self) -> dict[str, str]:

        conscientiousness=random.choice(self.conscientiousness)
        openness=random.choice(self.openness)

        if conscientiousness == 'organized':
            if openness == 'conservative':
                conjunction = 'However'
            else:
                conjunction = 'Further'
        else:
            if openness == 'conservative':
                conjunction = 'Moreover'
            else:
                conjunction = 'However'

        return {
            'neuroticism': random.choice(self.neuroticism),
            'extraversion': random.choice(self.extraversion),
            'conscientiousness': conscientiousness,
            'openness': openness,
            'agreeableness': random.choice(self.agreeableness),
            'conjunction': conjunction,
        }


class PersonaAugmentation:
    """
    A class to generate random persona attributes for a patient.
    The attributes are:
        - personality traits: Big Five personality traits
        - gender
        - age
        - occupation
        - education
        - name

    Name is sampled from the top 100 most commen names
    """

    def __init__(self, seed: int = 96):

        self.set_seed(seed)

        self.male_names = json.load(open('../persona_data/male_names.json'))
        self.female_names = json.load(open('../persona_data/female_names.json'))

        self.male_name_list = list(self.male_names.keys())
        self.male_name_prob_list = list(self.male_names.values())
        self.female_name_list = list(self.female_names.keys()) 
        self.female_name_prob_list = list(self.female_names.values())

        self.occupation_data = json.load(open('../persona_data/occupation_to_education.json'))
        self.occupation_key_list = list(self.occupation_data.keys())
        self.occupation_name_to_key = {
            val['occupation']: key for key, val in self.occupation_data.items()
        }
        self.occupation_list = [ele['occupation'] for ele in self.occupation_data.values()]
        self.occupation_prob_list = [
            val['frequency'] for val in self.occupation_data.values()
        ]

        self.education_data = json.load(open('../persona_data/education_to_occupation.json'))
        self.education_list = list(self.education_data.keys())

        self.big_five_traits_generator = BigFivePersonality()
        self.big_five_template = Template(
            open('./bigfive_template.txt', 'r').read()
        )


    def assign_personality_traits(self) -> str:
        traits_dict = self.big_five_traits_generator()
        traits_desc = self.big_five_template.render(**traits_dict)
        return traits_desc


    def assign_gender(self):
        candidates = ['male', 'female']
        return random.choice(candidates)


    def assign_age(self):
        return np.random.randint(16, 65)


    def assign_occupation_education(
        self, 
        education: str = '',
        occupation: str = '',
    ) -> tuple[str, str]:

        education = education.lower()
        occupation = occupation.lower()

        if education and occupation:
            return occupation, education

        # if education is available, sample occupation conditioning on the given education
        if education: 
            if education in self.education_data:
                occupation_data = self.education_data[education]
            else:
                # do fuzzy matching
                matched_tup_list = process.extract(
                    education, self.education_list, scorer=fuzz.ratio
                )
                matched_tup_list = sorted(matched_tup_list, key=lambda x: x[1], reverse=True)
                education = matched_tup_list[0][0]
                occupation_data = self.education_data[education]

            occupation_list = [ele['occupation'] for ele in occupation_data]
            occupation_prob_list = [ele['probability'] for ele in occupation_data]
            occupation = np.random.choice(occupation_list, p=occupation_prob_list)

        else:
            if occupation:
                # do fuzzy matching to find the closest occupation name
                matched_tup_list = process.extractOne(
                    occupation, self.occupation_list, scorer=fuzz.ratio
                )
                occupation = matched_tup_list[0]
                occupation_key = self.occupation_name_to_key[occupation]
            else:
                # this is the case where both education and occupation are unknown
                occupation_key = np.random.choice(self.occupation_key_list, p=self.occupation_prob_list)

            occupation_dict = self.occupation_data[occupation_key]
            education_dist = occupation_dict['education_distribution']
            education_list = list(education_dist.keys())
            education_prob_list = list(education_dist.values())

            occupation = occupation_dict['occupation']
            education = np.random.choice(education_list, p=education_prob_list)

        return occupation, education


    def assign_name(self, gender: str) -> str:

        if gender == 'male':
            return np.random.choice(self.male_name_list, p=self.male_name_prob_list)
        else:
            return np.random.choice(self.female_name_list, p=self.female_name_prob_list)


    @staticmethod
    def set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)


def main():

    persona_augmentor = PersonaAugmentation(seed=96)

    persona_data = json.load(open('./persona_structured.json', 'r'))
    persona_hub_data = json.load(open('./persona.json', 'r'))

    for key, val in persona_data.items():

        update_age, update_gender, update_occupation, update_education = False, False, False, False

        if (persona_gender := val['gender']) == 'unknown':
            gender = persona_augmentor.assign_gender()
            update_gender = True
        else:
            gender = persona_gender

        if val['age'] == 'unknown':
            age = persona_augmentor.assign_age()
            update_age = True

        # the case where the occupation is unknown
        if (persona_occupation := val['occupation']) == 'unknown':
            # check if the education is known
            if (persona_education := val['education']) == 'unknown':
                # assign a random occupation
                occupation, education = persona_augmentor.assign_occupation_education()
                update_education = True
            else:
                # assign an occupation conditioning on the given education
                occupation, education = persona_augmentor.assign_occupation_education(
                    education=persona_education
                )
            update_occupation = True

        #the case where occupation is known but education is unknown
        else:
            occupation, education = persona_augmentor.assign_occupation_education(
                occupation=persona_occupation
            )
            update_education = True

        name = persona_augmentor.assign_name(gender=gender)

        traits = persona_augmentor.assign_personality_traits()

        persona_data[key]['name'] = name
        persona_data[key]['traits'] = traits

        if update_age:
            persona_data[key]['age'] = age
        if update_gender:
            persona_data[key]['gender'] = gender
        if update_education:
            persona_data[key]['education'] = education
        if update_occupation:
            persona_data[key]['occupation'] = occupation

        # add original persona profile from PersonaHub to entry
        persona_hub_profile = persona_hub_data[key].strip()

        if not persona_hub_profile.endswith('.'):
            persona_hub_profile += '.'

        persona_data[key]['persona_hub'] = persona_hub_profile
    
    with open('./persona_augmented.json', 'w') as f:
        json.dump(persona_data, f, indent=4)


if __name__ == "__main__":
    main()
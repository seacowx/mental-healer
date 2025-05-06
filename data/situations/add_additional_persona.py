import json
import random
import numpy as np


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
        self.extraversion = ['extravert', 'introvert']
        self.openness = ['conventional', 'unconventional']
        self.agreeableness = ['agreeable', 'antagonistic']


    def __call__(self) -> dict[str, str]:
        return {
            'neuroticism': random.choice(self.neuroticism),
            'conscientiousness': random.choice(self.conscientiousness),
            'extraversion': random.choice(self.extraversion),
            'openness': random.choice(self.openness),
            'agreeableness': random.choice(self.agreeableness)
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

        print(len(self.male_names))
        print(len(self.female_names))
        raise SystemExit()

        self.occupation_data = json.load(open('../persona_data/occupation_to_education.json'))
        self.education_data = json.load(open('../persona_data/education_to_occupation.json'))

        self.big_five_traits_generator = BigFivePersonality()


    def assign_personality_traits(self, big_five_traits: BigFivePersonality) -> str:
        ...


    def assign_gender(self):
        candidates = ['male', 'female']
        return random.choice(candidates)


    def assign_age(self):
        return np.random.randint(16, 65)


    def assign_occupation(self):
        ...


    def assign_education(self):
        ...


    def assign_name(self):
        ...


    @staticmethod
    def set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)


def main():

    persona_augmentor = PersonaAugmentation(seed=96)


if __name__ == "__main__":
    main()
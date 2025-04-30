"""
Remove persona profiles that are too nice. These profiles make cognitive reframing a trivial task and will interfere with model training.

Before Filtering: 200,000
After Filtering: 172,752
"""

import json
import string
import random
from tqdm import tqdm

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
nltk.download("punkt")
nltk.download('punkt_tab')


def generate_id():
    """
    Randomly generate a 10-digit ID
    """
    id_length = 10
    id_characters = string.ascii_letters + string.digits
    return ''.join(random.choice(id_characters) for _ in range(id_length))


def filter_persona(persona_profile: str, stemmer: PorterStemmer, nice_word_set: set) -> bool:
    """
    Remove persona that are too nice. For example, "A tolerant and supportive person".
    This type of persona make cognitive reframing a trivial task and will interfere with model training. 
    """

    persona_tokens = word_tokenize(persona_profile.lower())
    persona_tokens = [stemmer.stem(token) for token in persona_tokens]

    keep_persona = True
    if any(word in persona_tokens for word in nice_word_set):
        keep_persona = False

    return keep_persona


def main():

    random.seed(96)
    stemmer = PorterStemmer()

    NICE_WORDS = [
        "calm", "relaxed", "peaceful", "laid-back", "even-tempered", "chill", "unflappable", "serene", "placid",
        "unperturbed", "tolerant", "open-minded", "accepting", "broad-minded", "nonjudgmental", "liberal", "inclusive",
        "understanding", "patient", "empathetic", "friendly", "warm", "kind", "approachable", "affable", "amiable",
        "good-natured", "genial", "cordial", "sociable", "supportive", "helpful", "cooperative", "accommodating",
        "encouraging", "compassionate", "caring", "considerate", "nurturing", "reassuring", "easy-going", "flexible",
        "adaptable", "agreeable", "mellow", "low-maintenance", "undemanding", "forgiving", "modest", "humble",
        "lighthearted", "playful", "cheerful", "fun-loving", "jovial", "witty", "humorous", "buoyant", "breezy",
        "upbeat", "tranquil", "cool-headed", "composed", "level-headed", "stoic", "permissive", "nonchalant",
        "detached", "neighborly", "companionable", "congenial", "personable", "down-to-earth",
        "sympathetic", "agreeing", "peaceable", "unassuming", "unpretentious", "tactful", "diplomatic", "yielding",
        "carefree", "giddy", "effervescent", "bubbly", "zany",
    ]
    STEMMED_NICE_WORDS = set()
    for word in NICE_WORDS:
        STEMMED_NICE_WORDS.add(stemmer.stem(word))

    # read the persona data in jsonl format
    with open('../PersonaHub/persona.jsonl', 'r') as f:
        persona_data = [json.loads(line) for line in f]

    out_dict = {}
    for entry in tqdm(persona_data):
        persona = entry['persona']

        keep_persona = filter_persona(
            persona_profile=persona,
            stemmer=stemmer,
            nice_word_set=STEMMED_NICE_WORDS,
        )

        if keep_persona:
            cur_id = generate_id()
            while cur_id in out_dict:
                cur_id = generate_id()
            out_dict[cur_id] = persona

    with open('./persona.json', 'w') as f:
        json.dump(out_dict, f, indent=4)


if __name__ == "__main__":
    main()
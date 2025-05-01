"""
Remove persona profiles that are too nice. These profiles make cognitive reframing a trivial task and will interfere with model training.

Before Filtering: 200,000
After Filtering: 172,752
"""

import json
import string
import random
from tqdm import tqdm

import torch
from jinja2 import Template
from vllm import LLM, SamplingParams

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
nltk.download("punkt")
nltk.download('punkt_tab')


def make_prompt(event_desc: str) -> str:

    template = Template(
        "Your task is to determine whether the given persona profile is describing an individual person or "
        "a group of people.\n\n<persona>\n{{ event_desc }}\n</persona>\n\n"
        "If the persona profile is describing an individual person, respond with '<decision>Individual</decision>'.\n"
        "If the persona profile is describing a group of people, respond with '<decision>Group</decision>'.\n"
        "Do not include any other text in your response."
    )

    return template.render(event_desc=event_desc.capitalize())


def parse_output(output):
    output = output.outputs[0].text
    if '<decision>' in output and '</decision>' in output:
        decision = output.split('<decision>')[1].split('</decision>')[0].strip().lower()
        return decision
    return ''


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

    temp_out_dict = {}
    for entry in tqdm(persona_data):
        persona = entry['persona']

        keep_persona = filter_persona(
            persona_profile=persona,
            stemmer=stemmer,
            nice_word_set=STEMMED_NICE_WORDS,
        )

        if keep_persona:
            cur_id = generate_id()
            while cur_id in temp_out_dict:
                cur_id = generate_id()
            temp_out_dict[cur_id] = persona

    # remove persona profiles that are not describing an individual
    model_path = (
        '/scratch/prj/charnu/seacow_hf_cache/models--Qwen--Qwen3-32B/'
        'snapshots/ba1f828c09458ab0ae83d42eaacc2cf8720c7957'
    )
    WORLD_SIZE = torch.cuda.device_count()
    vllm = LLM(
        model=model_path, 
        max_model_len=2048,
        tensor_parallel_size=WORLD_SIZE,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=8192,
        presence_penalty=1.,
    )

    prompt_list = [
        make_prompt(ele) for ele in temp_out_dict.values()
    ]
    msg_list = [
        [{'role': 'user', 'content': ele}] for ele in prompt_list
    ]

    print(msg_list[0]['content'])
    raise SystemExit()

    output_list = vllm.chat(
        messages=msg_list,
        sampling_params=sampling_params,
        use_tqdm=True,
        chat_template_kwargs={
            "enable_thinking": True,
        },
    )

    output_list = [
        parse_output(output)
        for output in output_list
    ]

    out_dict = {}
    for output_label, (key, val) in zip(output_list, temp_out_dict.items()):
        if output_label.lower() == 'individual':
            out_dict[key] = val

    print(f"Before Filtering: {len(temp_out_dict)}")
    print(f"After Filtering: {len(out_dict)}")

    with open('./persona.json', 'w') as f:
        json.dump(out_dict, f, indent=4)


if __name__ == "__main__":
    main()
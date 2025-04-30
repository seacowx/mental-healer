import json
import string
import random


def generate_id():
    """
    Randomly generate a 10-digit ID
    """
    id_length = 10
    id_characters = string.ascii_letters + string.digits
    return ''.join(random.choice(id_characters) for _ in range(id_length))


def main():
    random.seed(96)

    # read the persona data in jsonl format
    with open('../PersonaHub/persona.jsonl', 'r') as f:
        persona_data = [json.loads(line) for line in f]

    out_dict = {}
    for entry in persona_data:
        persona = entry['persona']
        cur_id = generate_id()
        while cur_id in out_dict:
            cur_id = generate_id()
        out_dict[cur_id] = persona

    with open('./persona.json', 'w') as f:
        json.dump(out_dict, f, indent=4)


if __name__ == "__main__":
    main()
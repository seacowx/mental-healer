import ast
import json
import random
import string


def generate_id():
    """
    Randomly generate a 10-digit ID
    """
    id_length = 10
    id_characters = string.ascii_letters + string.digits
    return ''.join(random.choice(id_characters) for _ in range(id_length))


def main():

    random.seed(96)
    data = open('./augesc.txt', 'r').readlines()

    data = [line.strip() for line in data if line.strip()]
    data = [ast.literal_eval(line) for line in data]

    init_dialogue_data = [
        ele[0] for ele in data
    ]

    out_dict = {}
    used_ids = set()
    for ele in init_dialogue_data:
        situation = ele[1]
        cur_id = generate_id()
        while cur_id in used_ids:
            cur_id = generate_id()
        used_ids.add(cur_id)
        out_dict[cur_id] = situation
    

    with open('./augesc.json', 'w') as f:
        json.dump(out_dict, f, indent=4)

if __name__ == "__main__":
    main()
import ast
import json


def main():

    data = open('./augesc.txt', 'r').readlines()

    data = [line.strip() for line in data if line.strip()]
    data = [ast.literal_eval(line) for line in data]

    init_dialogue_data = [
        ele[0] for ele in data
    ]

    with open('./augesc_full.json', 'w') as f:
        json.dump(data, f, indent=4)

    with open('./augesc.json', 'w') as f:
        json.dump(init_dialogue_data, f, indent=4)

if __name__ == "__main__":
    main()
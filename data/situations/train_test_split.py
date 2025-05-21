import json
import random

def main():

    random.seed(96)
    
    main_data = json.load(open('./situations_with_initial_thought_top5.json', 'r'))

    unique_situation_ids = list(main_data.keys())
    unique_situation_ids = list(set([
        ele.split('||')[0] for ele in unique_situation_ids
    ]))

    # sample 2000 situations as test set
    test_set_situation_ids = random.sample(unique_situation_ids, 2000)

    print(f'Test situation size: {len(test_set_situation_ids)}')
    print(f'Train situation size: {len(unique_situation_ids) - len(test_set_situation_ids)}')

    train_data, test_data = {}, {}
    for key, val in main_data.items():
        if key.split('||')[0] in test_set_situation_ids:
            test_data[key] = val
        else:
            train_data[key] = val

    # save test and train set
    print(len(train_data), len(test_data))

    with open('./situation_train.json', 'w') as f:
        json.dump(train_data, f, indent=4)

    with open('./situation_test.json', 'w') as f:
        json.dump(test_data, f, indent=4)


if __name__ == "__main__":
    main()

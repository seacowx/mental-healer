import json, yaml

from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score


def test_sentiment(sentiment_reward_model):

    test_data = json.load(open('../reward_ft/reward-sentiment/sentiment_data/reward-sentiment_test.json'))
    sentiment_label_mapping = yaml.safe_load(open('./configs/emotion_to_sentiment.yaml'))

    input_msg_list = [
        [{'role': 'user', 'content': ele['instruction'].strip()}]
        for ele in test_data
    ]
    label_list = [
        ele['output'].split('<sentiment>')[1].split('</sentiment>')[0].strip().lower()
        for ele in test_data
    ]

    outputs = sentiment_reward_model.get_sentiment(
        input_msg_list
    )

    # clean up the output, noise will cause issue in the label encoder

    # evaluate sentiment (coarse-grained)
    le = LabelEncoder()
    label_list = [sentiment_label_mapping[ele] for ele in label_list]
    label_encoded = le.fit_transform(label_list)
    output_encoded = le.transform(outputs)

    print(f"Accuracy: {accuracy_score(label_encoded, output_encoded)}, F1: {f1_score(label_encoded, output_encoded, average='weighted')}")

    raise SystemExit


def main():

    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B')
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-4B')

    input_msg = [
        {'role': 'user', 'content': 'How many "r"\'s are there in the word "strawberry"?'}
    ]
    input_chat_msg = tokenizer.apply_chat_template(
        input_msg, 
        add_generation_prompt=True,
        return_tensors='pt',
    )

    print(input_chat_msg)

    print(
        tokenizer.decode(input_chat_msg.input_ids[0])
    )
    raise SystemExit


if __name__ == '__main__':
    main()
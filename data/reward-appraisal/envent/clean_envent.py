"""
Code to remove envent entries where the emotion label is directly mentioned in the situation
"""

import json
from nltk.stem.porter import PorterStemmer  


def main():

    # Load the NLTK WordNet lemmatizer for lemmatizing emotion labels
    stemmer = PorterStemmer()

    original_envent_data = json.load(open("./envent_test.json"))

    cleaned_envent_data = {}
    for entry_id, entry in original_envent_data.items():

        situation = entry['context']
        stemmed_situation = ' '.join(
            [stemmer.stem(ele.lower()) for ele in situation.split() if ele.strip()]
        )

        emotion_label = entry['emotion_label']
        stemmed_emotion_label = stemmer.stem(emotion_label.lower())
        
        # Check if the stemmed emotion label is present in the stemmed stemmed_situation
        if stemmed_emotion_label in stemmed_situation:
            # Skip this entry
            continue

        # add the entry to the cleaned data
        cleaned_envent_data[entry_id] = entry


    print(f"Original ENVENT data size: {len(original_envent_data)}")
    print(f"Cleaned ENVENT data size: {len(cleaned_envent_data)}")

    # Save the cleaned data to a new JSON file
    with open("./envent_test_cleaned.json", "w") as f:
        json.dump(cleaned_envent_data, f, indent=4)


if __name__ == "__main__":
    main()

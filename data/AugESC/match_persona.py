import json
import torch
from transformers import AutoModel


def main():

    embedding_model = AutoModel.from_pretrained(
        "jinaai/jina-embeddings-v3", 
        device_map=0,
        trust_remote_code=True,
    )

    # load persona data
    with open('../PersonaHub/persona.jsonl', 'r') as f:
        persona_data = [json.loads(line) for line in f]

    persona_list = [
        ele['persona'] for ele in persona_data
    ]

    # load filtered situations
    with open('./augesc_content_filtered.json', 'r') as f:
        situation_data = json.load(f)

    # TODO: load filtered situations and match each situation with top-10 personas
    # print(f"Persona List Length: {len(persona_list)}")
    # embedded_persona = embedding_model.encode(
    #     persona_list, 
    #     task="retrieval.passage",
    #     show_progress_bar=True,
    # )
    
    embedded_situation = embedding_model.encode(
        list(situation_data.values()), 
        task="retrieval.query",
        show_progress_bar=True,
    )

    print(type(embedded_situation))
    
    # # compute cosine similarity between embedded_situation and embedded_persona
    # similarity_mtx = embedded_situation @ embedded_persona.T
    #
    # print(embedded_persona.shape)
    # print(embedded_situation.shape)
    # print(similarity_mtx.shape)


if __name__ == "__main__":
    main()

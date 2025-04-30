import json
from transformers import AutoModel


def main():

    embedding_model = AutoModel.from_pretrained(
        "jinaai/jina-embeddings-v3", 
        device_map=0,
        trust_remote_code=True,
    )

    # load jsonl
    with open('../PersonaHub/persona.jsonl', 'r') as f:
        persona_data = [json.loads(line) for line in f]

    persona_list = [
        ele['persona'] for ele in persona_data
    ]

    print(f"Persona List Length: {len(persona_list)}")

    embedded_persona = embedding_model.encode(
        persona_list, 
        task="retrieval.passage",
        show_progress_bar=True,
    )

    # TODO: load filtered situations and match each situation with top-10 personas
    print(embedded_persona.shape)


if __name__ == "__main__":
    main()

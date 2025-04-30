import json
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def main():

    embedding_model = AutoModel.from_pretrained(
        "jinaai/jina-embeddings-v3", 
        device_map=0,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "jinaai/jina-embeddings-v3"
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
    embedded_persona = embedding_model.encode(
        persona_list, 
        task="retrieval.passage",
        show_progress_bar=True,
    )
    embedded_persona = torch.tensor(embedded_persona).to(device=0)
    
    situation_list = list(situation_data.values())
    embedded_situation = embedding_model.encode(
        situation_list, 
        task="retrieval.query",
        show_progress_bar=True,
    )
    embedded_situation = torch.tensor(embedded_situation).to(device=0)
    print(type(embedded_situation))
    
    # compute cosine similarity between embedded_situation and embedded_persona
    similarity_mtx = embedded_situation @ embedded_persona.T

    # get the top-10 most similar personas for each situation
    top_10_indices = torch.topk(
        similarity_mtx, 
        k=10, 
        dim=1, 
        largest=True, 
        sorted=True
    ).indices

    print(top_10_indices.shape)


if __name__ == "__main__":
    main()

import json
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer



def main():

    embedding_model = AutoModel.from_pretrained(
        "jinaai/jina-embeddings-v3", 
        device_map=0,
        trust_remote_code=True,
    )

    # load persona data
    with open('../PersonaHub/persona.json', 'r') as f:
        persona_data = json.load(f)

    persona_list = list(persona_data.values())
    persona_id_list = list(persona_data.keys())

    # load filtered situations
    with open('./augesc_content_filtered.json', 'r') as f:
        situation_data = json.load(f)

    situation_id_list = list(situation_data.keys())

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

    top_10_indices = top_10_indices.cpu().numpy().tolist()

    # select the ids of the top-10 most similar personas for each situation
    top_10_persona_ids = {}
    for i in range(len(top_10_indices)):
        cur_situation_id = situation_id_list[i]
        top_10_persona_ids[cur_situation_id] = [persona_id_list[j] for j in top_10_indices[i]]

    persona_count = defaultdict(int)
    for indices in top_10_indices:
        for idx in indices:
            persona_count[persona_id_list[idx]] += 1

    with open('./augsec_matched_persona.json', 'w') as f:
        json.dump(top_10_persona_ids, f, indent=4)

    with open('./persona_count.json', 'w') as f:
        json.dump(persona_count, f, indent=4)


if __name__ == "__main__":
    main()

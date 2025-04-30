import json
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def mean_pooling(model_output, attention_mask):
    """
    Mean Pooling function from 
    https://huggingface.co/jinaai/jina-embeddings-v3
    """
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


@torch.no_grad()
def get_embedding(
    model, 
    tokenizer, 
    input_list
):

    encoded_input_list = tokenizer(
        input_list,
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    ).to(model.device) 
    task_id = model._adaptation_map['retrieval.query']
    adapter_mask = torch.full((len(input_list),), task_id, dtype=torch.int32)

    model_output = model(
        **encoded_input_list, 
        adapter_mask=adapter_mask.to(model.device),
    )

    embeddings = mean_pooling(model_output, encoded_input_list["attention_mask"])
    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings



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
    # print(f"Persona List Length: {len(persona_list)}")
    # embedded_persona = embedding_model.encode(
    #     persona_list, 
    #     task="retrieval.passage",
    #     show_progress_bar=True,
    # )
    
    situation_list = list(situation_data.values())
    embedded_situation = get_embedding(
        model=embedding_model,
        tokenizer=tokenizer,
        input_list=situation_list
    )
    print(type(embedded_situation))
    
    # # compute cosine similarity between embedded_situation and embedded_persona
    # similarity_mtx = embedded_situation @ embedded_persona.T
    #
    # print(embedded_persona.shape)
    # print(embedded_situation.shape)
    # print(similarity_mtx.shape)

    # get the top-10 most similar personas for each situation
    # top_10_indices = torch.topk(
    #     similarity_mtx, 
    #     k=10, 
    #     dim=1, 
    #     largest=True, 
    #     sorted=True
    # ).indices


if __name__ == "__main__":
    main()

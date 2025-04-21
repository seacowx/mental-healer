from transformers import AutoModel


class SemanticSimilarityReward:

    def __init__(self):
        self.embedding_model = AutoModel.from_pretrained(
            "jinaai/jina-embeddings-v3", 
            device_map=0,
            trust_remote_code=True,
        )


    def __get_embedding(self, text_list: list) -> list:
        return self.embedding_model.encode(
            text_list, 
            task="text-matching"
        )


    def compute_similarity(
        self, 
        utterance_list: list,
        response_list: list,
    ) -> list:

        utterance_embedded_mtx = self.__get_embedding(utterance_list) # N x D
        response_embedded_mtx = self.__get_embedding(response_list) # N x D

from src.config import (TOP_K,SIMILARITY_THRESHOLD)

from src.utils.logger import logger

class Retriever:

    def __init__(self,embedding_manager,vector_store):

        self.embedding_manager = (embedding_manager)

        self.vector_store = (vector_store)

    def retrieve(
        self,
        query,
        paper_id,
        top_k=TOP_K,
        threshold=SIMILARITY_THRESHOLD
    ):

        logger.info("Retrieving Documents...")

        query_embedding = (self.embedding_manager.embed_query(query))

        results = self.vector_store.search(
            query_embedding,
            top_k,paper_id
        )

        filtered_results = []

        for doc, meta, distance in zip(

            results["documents"][0],

            results["metadatas"][0],

            results["distances"][0]

        ):

            if distance <= threshold:

                filtered_results.append({

                    "document": doc,

                    "metadata": meta,

                    "distance": distance

                })

        logger.info(f"{len(filtered_results)} Relevant Chunks Retrieved.")

        return filtered_results
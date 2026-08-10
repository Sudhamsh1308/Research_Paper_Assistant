import chromadb

from src.config import (VECTOR_DB_DIR,COLLECTION_NAME)

from src.utils.logger import logger


class ChromaDBManager:

    def __init__(self):

        logger.info("Initializing ChromaDB...")

        self.client = chromadb.PersistentClient(
            path=str(VECTOR_DB_DIR)
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=COLLECTION_NAME
            )
        )

        logger.info("ChromaDB Ready.")

    def add_documents(
        self,
        ids,
        texts,
        embeddings,
        metadata
    ):

        logger.info(
            "Adding Documents..."
        )

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadata
        )

        logger.info(
            "Documents Added."
        )

    def search(self, query_embedding, top_k, paper_id=None):
        if paper_id:
            return self.collection.query(
            query_embeddings=[
                query_embedding.tolist()
            ],n_results=top_k,
            where={
                "paper_id": paper_id
            }
        )
        return self.collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=top_k)

    def count(self):

        return self.collection.count()

    def clear(self):
        count = self.collection.count()

        if count == 0:
            logger.info("ChromaDB is already empty.")
            return

        logger.info(f"Clearing {count} documents from ChromaDB...")

        ids = self.collection.get()["ids"]

        self.collection.delete(ids=ids)

        logger.info("ChromaDB Cleared.")
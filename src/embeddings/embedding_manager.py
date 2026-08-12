from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL

from src.utils.logger import logger


class EmbeddingManager:

    def __init__(self):

        logger.info("Loading Embedding Model...")

        self.model = SentenceTransformer(EMBEDDING_MODEL,device="cpu")

        logger.info("Embedding Model Loaded.")

    def generate_embeddings(self, chunks):
        logger.info("Generating Embeddings...")

        # Original text that will be stored as the document
        texts = [chunk.page_content for chunk in chunks]

        # Text specifically created for semantic embedding
        embedding_texts = []

        for chunk in chunks:
            metadata = chunk.metadata
            title = metadata.get("title")
            subject = metadata.get("subject")
            section = metadata.get("section", "Unknown")
            embedding_text = f"""
Paper: {title}
Subject: {subject}
Section: {section}
                                
Content:
{chunk.page_content}
"""
            embedding_texts.append(embedding_text)

        # Generate embeddings from metadata-enriched text
        embeddings = self.model.encode(embedding_texts,convert_to_numpy=True)

        logger.info(
        f"Generated {len(embeddings)} embeddings.")

        # Return ORIGINAL texts and embeddings
        return texts, embeddings

    def embed_query(self, query):

        logger.info("Generating query embedding...")

        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            batch_size=32,
            show_progress_bar=False
        )

        return embedding
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (CHUNK_SIZE,CHUNK_OVERLAP)

from src.utils.logger import logger


class ChunkManager:

    def __init__(self,chunk_size=CHUNK_SIZE,chunk_overlap=CHUNK_OVERLAP):

        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)

        logger.info("Chunk Manager Initialized")

    def split_documents(self,documents):

        logger.info("Splitting Documents...")

        chunks = self.text_splitter.split_documents(documents)

        logger.info(f"Generated {len(chunks)} chunks.")

        return chunks
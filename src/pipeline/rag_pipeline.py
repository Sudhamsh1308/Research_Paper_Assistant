from src.utils.logger import logger


class RAGPipeline:

    def __init__(
        self,
        pdf_loader,
        cleaner,
        section_detector,
        chunker,
        embedding_manager,
        vector_store,
        retriever,
        prompt_builder,
        llm
    ):

        self.pdf_loader = pdf_loader
        self.cleaner = cleaner
        self.section_detector = section_detector
        self.chunker = chunker
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.llm = llm

        logger.info("RAG Pipeline Initialized.")

    # ==========================================
    # Document Ingestion
    # ==========================================

    def ingest(self, pdf_path,paper_id):

        logger.info(
            f"Starting ingestion: {pdf_path}"
        )

        # 1. Load PDF
        documents = self.pdf_loader.load(
            pdf_path
        )

        # 2. Clean documents
        documents = self.cleaner.clean_documents(
            documents
        )

        # * Detect sections
        documents = self.section_detector.detect_sections(documents)
        
        # 3. Create chunks
        chunks = self.chunker.split_documents(
            documents
        )

        # 4. Generate embeddings
        texts, embeddings = (
            self.embedding_manager.generate_embeddings(
                chunks
            )
        )

        # 5. Create IDs
        ids = [
            f"{paper_id}_chunk_{i+1}"
            for i in range(len(chunks))
        ]

        # 6. Extract metadata
        metadata = [
    {
        **chunk.metadata,
        "paper_id": paper_id
    }
    for chunk in chunks]


        # 7. Store in ChromaDB
        self.vector_store.add_documents(
            ids=ids,
            texts=texts,
            embeddings=embeddings,
            metadata=metadata
        )

        logger.info(
            "Document ingestion completed."
        )

        return {
            "pages": len(documents),
            "chunks": len(chunks),
        }

    # ==========================================
    # Question Answering
    # ==========================================
    def ask(self, query,paper_id):

        logger.info(
            f"Processing query: {query}"
        )

        # 1. Retrieve relevant chunks
        retrieved_chunks = (
            self.retriever.retrieve(query,paper_id)
        )

        # 2. Handle no relevant results
        if not retrieved_chunks:

            return {
                "answer": (
                    "I couldn't find relevant "
                    "information in the uploaded "
                    "research papers."
                ),
                "sources": []
            }
        # 3. Build prompt
        prompt = (
            self.prompt_builder.build_prompt(
                query,
                retrieved_chunks
            )
        )
        # 4. Generate answer
        answer = self.llm.generate(
            prompt
        )
        # 5. Prepare sources
        sources = [
    {
        "page": chunk["metadata"].get("page", 0) + 1,
        "section": chunk["metadata"].get("section"),
        "text": chunk["document"][:300]
    }

    for chunk in retrieved_chunks]

        logger.info(
            "Query processing completed."
        )

        return {
            "answer": answer,
            "sources": sources
        }
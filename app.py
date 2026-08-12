import streamlit as st
import os

from src.loaders.pdf_loader import PDFLoader
from src.preprocess.section_detector import SectionDetector
from src.preprocess.cleaner import TextCleaner
from src.preprocess.chunker import ChunkManager
from src.embeddings.embedding_manager import EmbeddingManager
from src.vectorstore.chromaDB_manager import ChromaDBManager
from src.retrieval.retriever import Retriever
from src.prompts.prompt_builder import PromptBuilder
from src.llm.gemini_client import GeminiClient
from src.pipeline.rag_pipeline import RAGPipeline


# ------------------ PAGE ------------------

st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Research Paper Assistant")
st.caption("Upload a research paper and ask questions about it.")


# ------------------ CREATE PIPELINE ------------------

@st.cache_resource
def create_pipeline():

    pdf_loader = PDFLoader()
    cleaner = TextCleaner()
    section_detector = SectionDetector()
    chunker = ChunkManager()
    embedding_manager = EmbeddingManager()
    vector_store = ChromaDBManager()

    retriever = Retriever(
        embedding_manager,
        vector_store
    )

    prompt_builder = PromptBuilder()
    llm = GeminiClient()

    return RAGPipeline(
        pdf_loader,
        cleaner,
        section_detector,
        chunker,
        embedding_manager,
        vector_store,
        retriever,
        prompt_builder,
        llm
    )


pipeline = create_pipeline()


# ------------------ SESSION STATE ------------------

if "processed" not in st.session_state:
    st.session_state.processed = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "papers" not in st.session_state:
    st.session_state.papers = {}

if "selected_paper" not in st.session_state:
    st.session_state.selected_paper = None

# ------------------ SIDEBAR ------------------

with st.sidebar:

    st.header("📄 Upload Papers")

    files = st.file_uploader(
        "Choose research papers",
        type="pdf",
        accept_multiple_files=True
    )

    # ------------------ PROCESS PAPERS ------------------

    if files:

        if st.button(
            "Process Papers",
            use_container_width=True
        ):

            with st.spinner("Processing papers..."):

                for file in files:

                    os.makedirs(
                        "uploads",
                        exist_ok=True
                    )

                    path = os.path.join(
                        "uploads",
                        file.name
                    )

                    with open(path, "wb") as f:
                        f.write(file.getbuffer())

                    result = pipeline.ingest(
                        path,
                        file.name
                    )

                    st.session_state.papers[
                        file.name
                    ] = result

            st.session_state.processed = True

            # Select first paper only if nothing is selected
            if st.session_state.selected_paper is None:
                st.session_state.selected_paper = files[0].name

            st.success(
                f"{len(files)} paper(s) processed!"
            )
            st.rerun()

    # ------------------ PAPER SELECTION ------------------

    if st.session_state.papers:

        st.divider()

        st.write("### 📚 Your Papers")

        selected = st.selectbox(
            "Select a paper",
            list(st.session_state.papers.keys())
        )

        if selected != st.session_state.selected_paper:

            st.session_state.selected_paper = selected

            if selected not in st.session_state.chat_history:

                st.session_state.chat_history[selected] = []
                
                st.rerun()


    # ------------------ CLEAR CHAT ------------------

    if st.session_state.selected_paper:

        if st.button(
            "🗑️ Clear Chat",
            use_container_width=True
        ):

            st.session_state.chat_history[
                st.session_state.selected_paper
            ] = []

            st.rerun()


    # ------------------ MAIN AREA ------------------

if not st.session_state.papers:
    st.info(
        "Upload a research paper from the sidebar "
        "to start."
    )
else:
    st.subheader(
        f"📖 {st.session_state.selected_paper}"
    )

    st.divider()

    # ------------------ CURRENT CHAT ------------------

    paper_id = st.session_state.selected_paper

    if paper_id not in st.session_state.chat_history:
        st.session_state.chat_history[paper_id] = []

    messages = st.session_state.chat_history[paper_id]

    # ------------------ SUGGESTED QUESTIONS ------------------

    if not messages:
        st.write("### What would you like to know?")
        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "💡 What is the main idea?",
                use_container_width=True
            ):
                st.session_state.pending_question = (
                    "What is the main idea of this paper?"
                )

            if st.button(
                "📊 What are the main results?",
                use_container_width=True
            ):
                st.session_state.pending_question = (
                    "What are the main results of this paper?"
                )

        with col2:
            if st.button(
                "🔬 Explain the methodology",
                use_container_width=True
            ):
                st.session_state.pending_question = (
                "Explain the methodology used in this paper."
                )
            if st.button(
                "🎯 What are the key contributions?",
                use_container_width=True
            ):
                st.session_state.pending_question = (
                    "What are the key contributions of this paper?"
                )

        st.divider()


    # ------------------ CHAT INPUT ------------------

    question = st.chat_input(
        "Ask something about the paper...")

    # ------------------ SUGGESTED QUESTION ------------------

    if st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None

    # ------------------ PROCESS QUESTION ------------------

    if question:
        # Add user message
        messages.append({
        "role": "user",
        "content": question})

        # Generate answer
        with st.spinner("Thinking..."):
            try:
                response = pipeline.ask(
                question,
                st.session_state.selected_paper)

                answer = response.get(
                "answer",
                "I could not generate an answer.")

                sources = response.get(
                    "sources",
                    []
                )

            except Exception as e:
                answer = (
                f"❌ Error while generating answer:\n\n"
                f"`{str(e)}`"
            )

                sources = []

       # Add assistant message
        messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources})


    # ------------------ SHOW CHAT HISTORY ------------------
    
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander(
                    f"📑 Sources ({len(message['sources'])})"
                    ):
                    for source in message["sources"]:
                        st.markdown(
                            f"**📄 Page {source.get('page', 'Unknown')}** "
                            f"• **{source.get('section', 'Unknown')}**"
                        )
                        if source.get("text"):
                            st.caption(source["text"])
    
                        st.divider()

        # ------------------ IMPORTANT ------------------
        # DO NOT call st.rerun() here.
        #
        # Streamlit will rerun automatically when the
        # chat input/button interaction finishes.
        # The messages are already stored in session_state.
        # ------------------------------------------------
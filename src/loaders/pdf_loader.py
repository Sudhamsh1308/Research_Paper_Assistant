from langchain_community.document_loaders import PyMuPDFLoader

from utils.logger import logger



class PDFLoader:

    def __init__(self):

        logger.info("PDF Loader Initialized")

    def load(self, pdf_path):

        logger.info(f"Loading PDF : {pdf_path}")

        loader = PyMuPDFLoader(pdf_path)

        documents = loader.load()

        logger.info(
            f"Loaded {len(documents)} pages."
        )

        return documents

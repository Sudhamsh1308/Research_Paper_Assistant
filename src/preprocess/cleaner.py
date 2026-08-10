import re

from utils.logger import logger


class TextCleaner:

    def __init__(self):

        logger.info("Cleaner Initialized")

    def clean_text(self, text):

        text = text.replace("\r\n", "\n")

        text = text.replace("\t", " ")

        text = "\n".join(
            line.strip()
            for line in text.splitlines()
        )

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text
        )

        text = re.sub(
            r" {2,}",
            " ",
            text
        )

        return text.strip()

    def clean_documents(self, documents):

        logger.info(
            "Cleaning Documents..."
        )

        for doc in documents:

            doc.page_content = self.clean_text(
                doc.page_content
            )

        logger.info(
            "Cleaning Finished."
        )

        return documents
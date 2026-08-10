import re

from langchain_core.documents import Document
from src.utils.logger import logger


class SectionDetector:

    SECTION_NAMES = {
        "abstract",
        "introduction",
        "background",
        "related work",
        "literature review",
        "method",
        "methods",
        "methodology",
        "training",
        "results",
        "discussion",
        "conclusion",
        "references",
        "future work",
        "limitations",
        "model architecture"
    }

    def __init__(self):
        logger.info("Section Detector Initialized.")

    def detect_sections(self, documents):

        logger.info("Detecting sections...")

        result = []

        current_section = "Unknown"

        for document in documents:

            lines = document.page_content.splitlines()
            current_text = []

            for line in lines:

                line = line.strip()

                if not line:
                    continue

                section, remaining_text = self._detect_section(line)

                if section:

                    # Save previous text
                    if current_text:

                        metadata = dict(document.metadata)
                        metadata["section"] = current_section

                        result.append(
                            Document(
                                page_content="\n".join(current_text),
                                metadata=metadata
                            )
                        )

                        current_text = []
                        
                    current_section = section

                    if remaining_text:
                        current_text.append(remaining_text)

                else:
                    current_text.append(line)

            # Save remaining text
            if current_text:

                metadata = dict(document.metadata)
                metadata["section"] = current_section

                result.append(
                    Document(
                        page_content="\n".join(current_text),
                        metadata=metadata
                    )
                )

        logger.info(
            f"Section detection completed: {len(result)} segments."
        )

        return result

    def _detect_section(self, line):
        line = line.strip()
        line = re.sub(r"\s+", " ", line)

    # ---------------------------------------------
    # Abstract with content on same line
    #
    # Abstract—The dominant sequence...
    # Abstract: The dominant sequence...
    # ---------------------------------------------

        match = re.match(
        r"^abstract\s*[-–—:]\s*(.*)$",
        line,
        re.IGNORECASE
    )

        if match:
            remaining_text = match.group(1).strip()

            return "Abstract", remaining_text

    # ---------------------------------------------
    # Exact section name
    # ---------------------------------------------

        normalized = line.strip(" :.").lower()

        if normalized in self.SECTION_NAMES:
            return line.strip(" :."), ""

    # ---------------------------------------------
    # Numbered section
    # ---------------------------------------------

        match = re.match(
        r"^\d+(?:\.\d+)*\.?\s+(.+)$",
        line
    )

        if match:
            title = match.group(1).strip()

            if title.lower() in self.SECTION_NAMES:
                return title.title(), ""

    # ---------------------------------------------
    # Roman numeral section
    # ---------------------------------------------

        match = re.match(
        r"^[IVXLCDM]+\.?\s+(.+)$",
        line,
        re.IGNORECASE
    )

        if match:
            title = match.group(1).strip()

            if title.lower() in self.SECTION_NAMES:
                return title.title(), ""

        return None, ""
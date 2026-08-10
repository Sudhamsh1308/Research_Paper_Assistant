import os
import streamlit as st

from google import genai
from dotenv import load_dotenv

from src.config import GEMINI_MODEL
from src.utils.logger import logger


class GeminiClient:

    def __init__(self):

        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY")

        # For Streamlit Cloud
        if not api_key:
            api_key = st.secrets.get(
                "GEMINI_API_KEY",
                None
            )

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found."
            )

        self.client = genai.Client(
            api_key=api_key
        )

        self.model = GEMINI_MODEL

        logger.info(
            "Gemini Client Initialized."
        )


    def generate(self, prompt):

        logger.info(
            "Sending prompt to Gemini..."
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )

        logger.info(
            "Response received from Gemini."
        )

        return response.text
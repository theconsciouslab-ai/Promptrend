# llm_analyzer.py

import logging
import json
import re
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from agents.forum_agent.forum_config import (
    AZURE_ENDPOINT,
    AZURE_API_KEY,
    LLM_MODEL
    
)

logger = logging.getLogger("LLMAnalyzer")

class LLMAnalyzer:
    def __init__(self):
        self.client = ChatCompletionsClient(
            endpoint=AZURE_ENDPOINT,
            credential=AzureKeyCredential(AZURE_API_KEY)
        )
        self.model = LLM_MODEL

    def analyze(self, prompt: str) -> float:
        """
        Submit a prompt to the LLM and return a numeric score between 0.0 and 1.0.
        """
        try:
            response = self.client.complete(
                messages=[
                    SystemMessage(content="You are a security analyst specializing in LLM vulnerabilities. Respond only with a number between 0.0 and 1.0."),
                    UserMessage(content=prompt)
                ],
                temperature=0.0,
                max_tokens=100,
                model=self.model
            )
            content = response.choices[0].message.content.strip()
            
            # 🔍 Log the raw response from the model for debugging
            logger.warning(f"[LLM RAW OUTPUT] {content}")

            return self._extract_score(content)

        except Exception as e:
            logger.error(f"LLMAnalyzer error: {str(e)}")
            return 0.0


   

    def _extract_score(self, text: str) -> float:
        """
        Extract a float score between 0.0 and 1.0 from any text-like LLM output.

        Args:
            text (str): Raw LLM response

        Returns:
            float: Parsed score or 0.0 if invalid
        """
        try:
            # Extract the first number with optional decimal
            match = re.search(r"\b([01](?:\.?\d{1,3})?)\b", text)
            if match:
                value = float(match.group(1))
                return max(0.0, min(1.0, value))
            else:
                logger.warning(f"No float found in LLM response: '{text}'")
                return 0.0
        except Exception as e:
            logger.error(f"Failed to extract score: {str(e)} — LLM response: '{text}'")
            return 0.0

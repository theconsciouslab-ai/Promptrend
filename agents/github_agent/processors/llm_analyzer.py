# processors/llm_analyzer.py

import os
import logging
import asyncio
import random
import re
from datetime import datetime
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from  agents.github_agent.github_config import (
    LLM_MODEL,
    AZURE_ENDPOINT,
    AZURE_API_KEY,
    AZURE_MAX_REQUESTS_PER_MINUTE,
    AZURE_MAX_RETRIES,
    AZURE_CONCURRENCY,

)

logger = logging.getLogger("LLMAnalyzer")

class LLMAnalyzer:
    """
    Analyzes content using Azure-hosted DeepSeek-V3 to identify vulnerabilities.

    Uses Azure AI Inference SDK's ChatCompletionsClient for asynchronous, robust calls.
    """

    def __init__(self):
        """Initialize the LLM analyzer with API configuration and client."""
        # Load model and credentials
        self.model = LLM_MODEL  # e.g., "DeepSeek-V3"
        endpoint = AZURE_ENDPOINT.rstrip("/")
        credential = AzureKeyCredential(AZURE_API_KEY)

        # Rate limiting parameters
        self.request_count = 0
        self.last_reset = datetime.now()
        self.max_requests_per_minute = AZURE_MAX_REQUESTS_PER_MINUTE
        self.max_retries = AZURE_MAX_RETRIES
        self.semaphore = asyncio.Semaphore(AZURE_CONCURRENCY)


        # Initialize Azure AI Inference ChatCompletions client
        self.client = ChatCompletionsClient(endpoint, credential)

    async def analyze(self, prompt: str) -> float:
            """
            Evaluate the given prompt with DeepSeek-V3 and return a float score in [0.0, 1.0].
            """
            await self._check_rate_limit()
            system_msg = (
                "You are a security analysis assistant that evaluates content "
                "for LLM vulnerability indicators."
            )
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ]

            async with self.semaphore:
                for attempt in range(self.max_retries):
                    try:
                        response = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: self.client.complete(
                                model=self.model,
                                messages=messages,
                                temperature=0.0,
                                max_tokens=500
                            )
                        )
                        self.request_count += 1

                        choice = response.choices[0].message.content.strip()

                        # Extract float using regex
                        match = re.search(r"\b([01](?:\.\d+)?)\b", choice)
                        if match:
                            score = float(match.group(1))
                            return max(0.0, min(1.0, score))
                        else:
                            logger.error(f"❌ Could not extract float from response: '{choice}'")
                            return 0.0

                    except Exception as e:
                        msg = str(e)
                        if '429' in msg:
                            logger.warning("Rate limit hit; retrying...")
                        elif '500' in msg:
                            logger.warning("Server error; retrying...")
                        else:
                            logger.error(f"LLM API call error: {e}")
                            return 0.0

                        delay = self._calculate_backoff(attempt)
                        await asyncio.sleep(delay)

            logger.error(f"Exceeded max retries ({self.max_retries}); returning 0.0")
            return 0.0

    async def _check_rate_limit(self):
        """
        Throttle requests to avoid exceeding Azure rate limits.
        """
        now = datetime.now()
        elapsed = (now - self.last_reset).total_seconds()
        if elapsed >= 60:
            self.request_count = 0
            self.last_reset = now
            return

        if self.request_count >= (self.max_requests_per_minute * 0.8):
            wait_time = max(0, 60 - elapsed) + random.uniform(0, 1)
            logger.info(f"Approaching rate limit; sleeping {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
            self.request_count = 0
            self.last_reset = datetime.now()

    def _calculate_backoff(self, attempt: int) -> float:
        """
        Exponential backoff with jitter.
        """
        jitter = random.uniform(0, 1)
        return min(60, (2 ** attempt) + jitter)

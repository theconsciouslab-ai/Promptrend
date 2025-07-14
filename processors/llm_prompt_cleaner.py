# processors/llm_prompt_cleaner.py

import logging
import asyncio
import random
from datetime import datetime
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from data_collection_config import (
    AZURE_API_KEY, AZURE_ENDPOINT, LLM_MODEL,
    AZURE_MAX_REQUESTS_PER_MINUTE, AZURE_MAX_RETRIES, AZURE_CONCURRENCY
)

logger = logging.getLogger("LLMPromptCleaner")

class LLMPromptCleaner:
    def __init__(self):
        self.model = LLM_MODEL
        self.client = ChatCompletionsClient(
            endpoint=AZURE_ENDPOINT.rstrip("/"),
            credential=AzureKeyCredential(AZURE_API_KEY)
        )
        self.request_count = 0
        self.last_reset = datetime.now()
        self.max_requests_per_minute = AZURE_MAX_REQUESTS_PER_MINUTE
        self.max_retries = AZURE_MAX_RETRIES
        self.semaphore = asyncio.Semaphore(int(AZURE_CONCURRENCY or 3))

    async def clean_prompt(self, text: str) -> str | None:
        if not text.strip():
            return None

        system_message = SystemMessage(content=(
            "You are an adversarial prompt extractor. "
            "Extract only the actual prompt that tries to jailbreak, bypass, or override safety. "
            "Return only the cleaned prompt. If none exists, return: null"
        ))

        user_message = UserMessage(content=f"Text:\n{text.strip()}\n\nExtract the adversarial prompt.")

        await self._check_rate_limit()
        async with self.semaphore:
            for attempt in range(self.max_retries):
                try:
                    response = await asyncio.to_thread(
                        self.client.complete,
                        model=self.model,
                        messages=[system_message, user_message],
                        max_tokens=300,
                        temperature=0.2
                    )
                    result = response.choices[0].message.content.strip()
                    if result.lower() == "null":
                        return None
                    return result

                except Exception as e:
                    logger.warning(f"[LLMPromptCleaner] Retry {attempt + 1} failed: {e}")
                    await asyncio.sleep(self._calculate_backoff(attempt))

            logger.error("[LLMPromptCleaner] Max retries exceeded.")
            return None

    async def _check_rate_limit(self):
        now = datetime.now()
        elapsed = (now - self.last_reset).total_seconds()
        if elapsed >= 60:
            self.request_count = 0
            self.last_reset = now
        if self.request_count >= self.max_requests_per_minute * 0.8:
            sleep_time = max(0, 60 - elapsed) + random.uniform(0.1, 1.0)
            logger.info(f"[LLMPromptCleaner] Approaching rate limit. Sleeping for {sleep_time:.1f}s")
            await asyncio.sleep(sleep_time)
            self.request_count = 0
            self.last_reset = datetime.now()
        self.request_count += 1

    def _calculate_backoff(self, attempt: int) -> float:
        return min(60.0, (2 ** attempt) + random.uniform(0, 1))

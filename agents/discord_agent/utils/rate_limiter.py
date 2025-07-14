# utils/rate_limiter.py
import asyncio
import time
import logging
from agents.discord_agent.discord_config import (
    BOT_TOKEN,
TARGET_SERVERS,
LOG_FILE,
LOG_LEVEL,
DATA_DIR,
DEFAULT_REQUESTS_PER_SECOND,
DISCUSSION_SCORE_WEIGHT,
CODE_SCORE_WEIGHT,
ENABLE_PROMPT_EXTRACTION,
MONITOR_ALL_CHANNELS,
SECURITY_KEYWORDS,
TECHNICAL_KEYWORDS,
MODEL_KEYWORDS,
VULNERABILITY_KEYWORDS,
MIN_INLINE_CODE_LENGTH,
RELEVANT_URL_DOMAINS,
CODE_FILE_EXTENSIONS,
TEXT_FILE_EXTENSIONS,
COMMAND_PREFIX,
LDA_NUM_TOPICS,
LDA_MAX_DF,
LDA_MIN_DF,
LDA_TOPIC_MATCH_THRESHOLD,
LDA_TOPIC_SCORE_THRESHOLD,
CATEGORY_CONFIDENCE,
DEBUG_MODE,
RELEVANCE_THRESHOLD,
CONTEXT_WINDOW_SECONDS,
CONVERSATION_CACHE_TIMEOUT,
HISTORICAL_BACKFILL_DAYS,
HISTORICAL_BATCH_DELAY,
DISCORD_HISTORY_BATCH_SIZE,
LLM_MODEL,
LLM_PROVIDER,
MAX_RATE_LIMIT,
AZURE_ENDPOINT,
AZURE_API_VERSION,
AZURE_API_KEY,
LLM_MAX_RETRIES,
LLM_MAX_TOKENS,
LLM_TEMPERATURE,
LLM_SYSTEM_PROMPT,
    
)

logger = logging.getLogger("RateLimiter")

class RateLimiter:
    """
    Manages API rate limits for Discord API.
    
    Implements a token bucket algorithm to respect Discord's
    rate limiting policies and avoid getting throttled.
    """
    
    def __init__(self, requests_per_second=DEFAULT_REQUESTS_PER_SECOND):        
        """
        Initialize the rate limiter.
        
        Args:
            requests_per_second: Maximum number of requests per second
        """
        self.requests_per_second = requests_per_second
        self.tokens = requests_per_second
        self.last_replenish = time.time()
        self.lock = asyncio.Lock()
        
    async def wait_if_needed(self):
        """
        Wait if necessary to stay within rate limits.
        """
        async with self.lock:
            # Replenish tokens based on elapsed time
            now = time.time()
            elapsed = now - self.last_replenish
            self.tokens = min(
                self.requests_per_second,
                self.tokens + elapsed * self.requests_per_second
            )
            self.last_replenish = now
            
            # If we don't have a token, calculate wait time
            if self.tokens < 1:
                wait_time = (1 - self.tokens) / self.requests_per_second
                logger.debug(f"Rate limit hit, waiting {wait_time:.2f} seconds")
                
                # Release lock while waiting
                self.lock.release()
                await asyncio.sleep(wait_time)
                await self.lock.acquire()
                
                # Replenish again after waiting
                now = time.time()
                elapsed = now - self.last_replenish
                self.tokens = min(
                    self.requests_per_second,
                    self.tokens + elapsed * self.requests_per_second
                )
                self.last_replenish = now
            
            # Consume a token
            self.tokens -= 1
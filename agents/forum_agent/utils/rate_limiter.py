# utils/rate_limiter.py

import time
import asyncio
import logging

logger = logging.getLogger("RateLimiter")

class RateLimiter:
    """
    Simple rate limiter that enforces a minimum delay between requests.
    Useful for preventing overloading of target forums.
    """

    def __init__(self, delay: float = 1.0):
        """
        Initialize the rate limiter.

        Args:
            delay (float): Minimum delay in seconds between requests
        """
        self.delay = delay
        self._last_request_time = 0.0

    async def wait_if_needed(self):
        """
        Wait if the last request was too recent.
        """
        now = time.time()
        elapsed = now - self._last_request_time

        if elapsed < self.delay:
            wait_time = self.delay - elapsed
            logger.debug(f"Rate limiter sleeping for {wait_time:.2f} seconds")
            await asyncio.sleep(wait_time)

        self._last_request_time = time.time()

# utils/rate_limiter.py
import asyncio
import time
import logging

logger = logging.getLogger("RateLimiter")


class RateLimiter:
    """
    Simple async rate limiter to respect GitHub API limits.
    """

    def __init__(self, delay=1.0):
        """
        Args:
            delay: Minimum delay (in seconds) between requests.
        """
        self.delay = delay
        self.last_call = time.time()
        self._lock = asyncio.Lock()

    async def wait_if_needed(self):
        async with self._lock:
            elapsed = time.time() - self.last_call
            if elapsed < self.delay:
                sleep_time = self.delay - elapsed
                logger.debug(f"Rate limiter sleeping for {sleep_time:.2f} seconds")
                await asyncio.sleep(sleep_time)
            self.last_call = time.time()

# utils/ip_rotator.py

import random
import logging
from typing import Optional
from agents.forum_agent.forum_config import (
    USE_PROXY_ROTATION,
    PROXIES
)

logger = logging.getLogger("IPRotator")

class IPRotator:
    """
    Manages proxy rotation for HTTP requests.
    Useful for evading IP-based rate limiting or bans.
    """

    def __init__(self):
        self.use_rotation = USE_PROXY_ROTATION
        self.proxies = PROXIES
        self.current_proxy = None

    async def get_proxy(self) -> Optional[str]:
        """
        Get a proxy URL to use for the next request.

        Returns:
            str or None: Proxy URL or None if not using proxies
        """
        if not self.use_rotation or not self.proxies:
            return None

        # Use current proxy or select a random one
        if not self.current_proxy:
            self.current_proxy = random.choice(self.proxies)

        logger.debug(f"Using proxy: {self.current_proxy}")
        return self.current_proxy

    async def rotate(self):
        """
        Rotate to a new proxy.
        """
        if not self.use_rotation or not self.proxies:
            return

        old_proxy = self.current_proxy
        options = [p for p in self.proxies if p != old_proxy]

        if options:
            self.current_proxy = random.choice(options)
            logger.info(f"Rotated proxy from {old_proxy} to {self.current_proxy}")
        else:
            logger.warning("No alternative proxy available to rotate")

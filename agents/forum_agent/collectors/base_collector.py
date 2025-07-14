# collectors/base_collector.py
import logging
import asyncio
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, List, Any
from agents.forum_agent.forum_config import (
    USE_PROXY_ROTATION,
    HEADERS,
    REQUEST_TIMEOUT,
    RATE_LIMIT_DELAY
    
)

logger = logging.getLogger("BaseForumCollector")

class BaseForumCollector(ABC):
    """
    Abstract base class for forum collectors.
    
    Defines the interface that all forum-specific collectors must implement
    and provides common functionality for thread collection and processing.
    """
    
    def __init__(self, agent):
        """
        Initialize the base forum collector.
        
        Args:
            agent: The Discussion Forums agent instance
        """
        self.agent = agent
        self.rate_limiter = agent.rate_limiter
        self.ip_rotator = agent.ip_rotator
    
    @abstractmethod
    async def collect_threads(self, forum_url: str, lexicon: List[str], 
                             since_time: datetime, language: str) -> List[Dict[str, Any]]:
        """
        Collect relevant threads from a forum.
        
        Args:
            forum_url: URL of the forum
            lexicon: Language-specific keyword lexicon
            since_time: Time to collect threads from
            language: Forum language
            
        Returns:
            list: Collected thread data
        """
        pass
    
    @abstractmethod
    async def _extract_thread_structure(self, thread_url: str) -> Dict[str, Any]:
        """
        Extract the structure and content of a thread.
        
        Args:
            thread_url: URL of the thread
            
        Returns:
            dict: Thread data including content and structure
        """
        pass
    
    async def _check_relevance(self, text: str, title: str, lexicon: List[str]) -> bool:
        """
        Check if thread content is relevant based on lexicon.
        
        Args:
            text: Thread content
            title: Thread title
            lexicon: Keyword lexicon
            
        Returns:
            bool: True if relevant, False otherwise
        """
        # Check title first (higher weight)
        title_lower = title.lower()
        for keyword in lexicon:
            if keyword.lower() in title_lower:
                return True
        
        # Check content
        text_lower = text.lower()
        for keyword in lexicon:
            if keyword.lower() in text_lower:
                return True
                
        return False
    
    async def _get_with_retry(self, url: str, max_retries: int = 3) -> str:
        """
        Get URL content with retry logic and IP rotation.
        
        Args:
            url: URL to fetch
            max_retries: Maximum number of retry attempts
            
        Returns:
            str: HTML content or empty string on failure
        """
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Wait for rate limit if needed
                await self.rate_limiter.wait_if_needed()
                
                # Get a proxy or direct connection
                proxy = await self.ip_rotator.get_proxy() if USE_PROXY_ROTATION else None
                
                # Make request
                async with self.agent.session.get(
                    url, 
                    headers=HEADERS,
                    proxy=proxy,
                    timeout=REQUEST_TIMEOUT
                ) as response:
                    if response.status == 200:
                        return await response.text()
                    
                    if response.status == 429:  # Too many requests
                        retry_count += 1
                        logger.warning(f"Rate limited for {url}, rotating IP and retrying ({retry_count}/{max_retries})")
                        await asyncio.sleep(RATE_LIMIT_DELAY * retry_count)
                        await self.ip_rotator.rotate()
                        continue
                        
                    if response.status == 403:  # Forbidden
                        logger.warning(f"Access forbidden for {url}, rotating IP and retrying")
                        await self.ip_rotator.rotate()
                        retry_count += 1
                        continue
                        
                    logger.warning(f"Failed to fetch {url}: Status {response.status}")
                    return ""
                    
            except asyncio.TimeoutError:
                logger.warning(f"Timeout fetching {url}, retrying ({retry_count+1}/{max_retries})")
                retry_count += 1
                continue
                
            except Exception as e:
                logger.error(f"Error fetching {url}: {str(e)}")
                return ""
                
        logger.error(f"Max retries exceeded for {url}")
        return ""
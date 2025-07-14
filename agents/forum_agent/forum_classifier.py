# forum_classifier.py
import logging
import re
import aiohttp
from bs4 import BeautifulSoup
from typing import Dict, Any, Tuple
from agents.forum_agent.forum_config import (
    MULTILINGUAL_LEXICONS,
    HEADERS,
)

logger = logging.getLogger("ForumClassifier")

class ForumClassifier:
    """
    Classifies forums based on platform type and security relevance.
    
    This component identifies the underlying forum platform software 
    and evaluates the relevance of a forum for LLM vulnerability monitoring.
    """
    
    def __init__(self):
        """Initialize the forum classifier."""
        # Platform detection patterns
        self.platform_patterns = {
            "vbulletin": [
                r'<meta name="generator" content="vBulletin',
                r'powered by vBulletin',
                r'vb_[a-zA-Z_]+'
            ],
            "discourse": [
                r'<meta name="generator" content="Discourse',
                r'Powered by Discourse',
                r'DiscourseBoot'
            ],
            "phpbb": [
                r'<meta name="generator" content="phpBB',
                r'Powered by phpBB',
                r'phpbb[a-zA-Z_]+'
            ],
            "xenforo": [
                r'<html id="XenForo"',
                r'<body class="xenforo',
                r'Powered by XenForo'
            ]
        }
        
        # Relevance keywords per language
        self.relevance_keywords = MULTILINGUAL_LEXICONS
        
    async def detect_forum_platform(self, url: str) -> str:
        """
        Detect the underlying forum platform software.
        
        Args:
            url: Forum URL
            
        Returns:
            str: Detected platform type or "custom" if unknown
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=HEADERS) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to access {url}: Status {response.status}")
                        return "custom"
                        
                    html = await response.text()
                    
                    # Check for platform patterns in HTML
                    for platform, patterns in self.platform_patterns.items():
                        for pattern in patterns:
                            if re.search(pattern, html, re.IGNORECASE):
                                logger.info(f"Detected {platform} forum at {url}")
                                return platform
                    
                    logger.info(f"Could not determine platform for {url}, using custom collector")
                    return "custom"
                    
        except Exception as e:
            logger.error(f"Error detecting forum platform for {url}: {str(e)}")
            return "custom"
    
    async def assess_forum_relevance(self, url: str, language: str = 'en') -> float:
        """
        Assess the relevance of a forum for LLM vulnerability monitoring.
        
        Args:
            url: Forum URL
            language: Primary language of the forum
            
        Returns:
            float: Relevance score between 0.0 and 1.0
        """
        try:
            # Get language-specific keywords
            keywords = self.relevance_keywords.get(language, self.relevance_keywords['en'])
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=HEADERS) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to access {url}: Status {response.status}")
                        return 0.0
                        
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract text content
                    text = soup.get_text().lower()
                    
                    # Count keyword occurrences
                    keyword_matches = 0
                    for keyword in keywords:
                        keyword_matches += text.count(keyword.lower())
                    
                    # Calculate relevance score based on keyword density
                    word_count = len(text.split())
                    if word_count > 0:
                        keyword_density = min(1.0, keyword_matches / (word_count * 0.01))
                        return keyword_density * 0.7  # Scale to reasonable range
                    
                    return 0.0
                    
        except Exception as e:
            logger.error(f"Error assessing forum relevance for {url}: {str(e)}")
            return 0.0
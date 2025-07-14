# collectors/vbulletin_collector.py
import logging
import re
import asyncio
from datetime import datetime , timezone
from typing import Dict, List, Any
from bs4 import BeautifulSoup

from agents.forum_agent.collectors.base_collector import BaseForumCollector
from agents.forum_agent.forum_config import (
    INTER_PAGE_DELAY,
    
)

logger = logging.getLogger("VBulletinCollector")

class VBulletinCollector(BaseForumCollector):
    """
    Collector for vBulletin-based forums.
    
    Implements forum-specific collection logic for vBulletin,
    a popular forum software used by many security communities.
    """
    
    async def collect_threads(self, forum_url: str, lexicon: List[str], 
                             since_time: datetime, language: str) -> List[Dict[str, Any]]:
        """
        Collect relevant threads from a vBulletin forum.
        
        Args:
            forum_url: URL of the forum
            lexicon: Language-specific keyword lexicon
            since_time: Time to collect threads from
            language: Forum language
            
        Returns:
            list: Collected thread data
        """
        logger.info(f"Collecting threads from vBulletin forum: {forum_url}")
        
        threads = []
        forum_name = self._extract_forum_name(forum_url)
        
        try:
            # Get forum sections (typically in format /forumdisplay.php?f=X)
            sections = await self._get_forum_sections(forum_url)
            
            for section_url in sections:
                # Get thread listing
                thread_urls = await self._get_thread_listing(section_url)
                
                # Process each thread
                for thread_url in thread_urls:
                    # Get thread data
                    thread_data = await self._extract_thread_structure(thread_url)
                    
                    if not thread_data:
                        continue
                    
                    # Add forum metadata
                    thread_data['forum_id'] = self._extract_forum_id(forum_url)
                    thread_data['forum_name'] = forum_name
                    thread_data['forum_url'] = forum_url
                    
                    # Check thread date if available
                    thread_date = thread_data.get('date')
                    if thread_date and thread_date < since_time:
                        continue
                    
                    # Check relevance
                    is_relevant = await self._check_relevance(
                        thread_data.get('content', ''),
                        thread_data.get('title', ''),
                        lexicon
                    )
                    
                    if is_relevant:
                        threads.append(thread_data)
                
                # Sleep between sections to avoid detection
                await asyncio.sleep(INTER_PAGE_DELAY)
            
            logger.info(f"Collected {len(threads)} relevant threads from {forum_url}")
            
        except Exception as e:
            logger.error(f"Error collecting vBulletin threads from {forum_url}: {str(e)}")
            
        return threads
    
    async def _extract_thread_structure(self, thread_url: str) -> Dict[str, Any]:
        """
        Extract the structure and content of a vBulletin thread.
        
        Args:
            thread_url: URL of the thread
            
        Returns:
            dict: Thread data including content and structure
        """
        try:
            html = await self._get_with_retry(thread_url)
            if not html:
                return {}
                
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract thread title
            title_element = soup.select_one('.threadtitle')
            title = title_element.get_text().strip() if title_element else "Untitled Thread"
            
            # Extract thread date
            date_element = soup.select_one('.postdate')
            date_text = date_element.get_text().strip() if date_element else ""
            thread_date = self._parse_date(date_text)
            
            # Extract thread author
            author_element = soup.select_one('.username')
            author = author_element.get_text().strip() if author_element else "Unknown"
            
            # Extract thread content (first post)
            content_element = soup.select_one('.postcontent')
            content = content_element.get_text().strip() if content_element else ""
            
            # Extract all posts in thread
            posts = []
            post_elements = soup.select('.postcontainer')
            
            for post_element in post_elements:
                post_author_element = post_element.select_one('.username')
                post_author = post_author_element.get_text().strip() if post_author_element else "Unknown"
                
                post_content_element = post_element.select_one('.postcontent')
                post_content = post_content_element.get_text().strip() if post_content_element else ""
                
                post_date_element = post_element.select_one('.postdate')
                post_date_text = post_date_element.get_text().strip() if post_date_element else ""
                post_date = self._parse_date(post_date_text)
                
                posts.append({
                    'author': post_author,
                    'content': post_content,
                    'date': post_date
                })
            
            # Combine all posts for analysis
            all_content = "\n\n".join([p.get('content', '') for p in posts])
            
            # Create thread data
            thread_data = {
                'thread_id': self._extract_thread_id(thread_url),
                'url': thread_url,
                'title': title,
                'author': author,
                'date': thread_date,
                'content': all_content,
                'posts': posts
            }
            
            return thread_data
            
        except Exception as e:
            logger.error(f"Error extracting vBulletin thread structure from {thread_url}: {str(e)}")
            return {}
    
    async def _get_forum_sections(self, forum_url: str) -> List[str]:
        """
        Get forum section URLs.
        
        Args:
            forum_url: Base forum URL
            
        Returns:
            list: Section URLs
        """
        sections = []
        
        try:
            html = await self._get_with_retry(forum_url)
            if not html:
                return []
                
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find forum sections (typical vBulletin structure)
            section_links = soup.select('a.forumtitle')
            
            for link in section_links:
                href = link.get('href', '')
                if href and 'forumdisplay.php' in href:
                    # Convert to absolute URL if needed
                    if href.startswith('/'):
                        parts = forum_url.split('/')
                        base_url = '/'.join(parts[:3])  # http(s)://domain.com
                        section_url = base_url + href
                    elif not href.startswith('http'):
                        section_url = forum_url.rstrip('/') + '/' + href
                    else:
                        section_url = href
                        
                    sections.append(section_url)
            
            # If no sections found, use the forum URL as the only section
            if not sections:
                sections.append(forum_url)
                
        except Exception as e:
            logger.error(f"Error getting vBulletin forum sections from {forum_url}: {str(e)}")
            
        return sections
    
    async def _get_thread_listing(self, section_url: str) -> List[str]:
        """
        Get thread URLs from a forum section.
        
        Args:
            section_url: Section URL
            
        Returns:
            list: Thread URLs
        """
        thread_urls = []
        
        try:
            html = await self._get_with_retry(section_url)
            if not html:
                return []
                
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find thread links (typical vBulletin structure)
            thread_links = soup.select('a.title')
            
            for link in thread_links:
                href = link.get('href', '')
                if href and ('showthread.php' in href or 'thread' in href):
                    # Convert to absolute URL if needed
                    if href.startswith('/'):
                        parts = section_url.split('/')
                        base_url = '/'.join(parts[:3])  # http(s)://domain.com
                        thread_url = base_url + href
                    elif not href.startswith('http'):
                        thread_url = section_url.rstrip('/').rsplit('/', 1)[0] + '/' + href
                    else:
                        thread_url = href
                        
                    thread_urls.append(thread_url)
                
        except Exception as e:
            logger.error(f"Error getting vBulletin thread listing from {section_url}: {str(e)}")
            
        return thread_urls
    
    def _extract_forum_id(self, forum_url: str) -> str:
        """
        Extract forum ID from URL.
        
        Args:
            forum_url: Forum URL
            
        Returns:
            str: Forum ID
        """
        # Try to extract forum ID from URL
        match = re.search(r'f=(\d+)', forum_url)
        if match:
            return match.group(1)
            
        # Use domain as fallback
        match = re.search(r'://([^/]+)', forum_url)
        if match:
            return match.group(1)
            
        return "unknown"
    
    def _extract_forum_name(self, forum_url: str) -> str:
        """
        Extract forum name from URL or content.
        
        Args:
            forum_url: Forum URL
            
        Returns:
            str: Forum name
        """
        # Extract domain as forum name
        match = re.search(r'://([^/]+)', forum_url)
        if match:
            domain = match.group(1)
            # Clean up domain (remove www. and .com/.org/etc)
            domain = domain.replace('www.', '')
            domain = re.sub(r'\.(com|org|net|io|co|me|info)$', '', domain)
            return domain.title()
            
        return "Unknown Forum"
    
    def _extract_thread_id(self, thread_url: str) -> str:
        """
        Extract thread ID from URL.
        
        Args:
            thread_url: Thread URL
            
        Returns:
            str: Thread ID
        """
        # Try to extract thread ID from URL
        match = re.search(r't=(\d+)', thread_url)
        if match:
            return match.group(1)
            
        # Alternative format
        match = re.search(r'threadid=(\d+)', thread_url)
        if match:
            return match.group(1)
            
        # Use entire URL as fallback
        return thread_url
    
    def _parse_date(self, date_text: str) -> datetime:
        """
        Parse date from vBulletin date format.
        
        Args:
            date_text: Date text from forum
            
        Returns:
            datetime: Parsed date or current date if parsing fails
        """
        try:
            # vBulletin has various date formats
            date_formats = [
                '%m-%d-%Y, %I:%M %p',  # 01-25-2023, 10:30 AM
                '%d-%m-%Y, %I:%M %p',  # 25-01-2023, 10:30 AM
                '%Y-%m-%d, %I:%M %p',  # 2023-01-25, 10:30 AM
                '%B %d, %Y, %I:%M %p',  # January 25, 2023, 10:30 AM
                '%d %B %Y, %I:%M %p',  # 25 January 2023, 10:30 AM
                '%a %b %d, %Y %I:%M %p',  # Wed Jan 25, 2023 10:30 AM
                '%a, %d %b %Y %H:%M:%S +0000',  # Wed, 25 Jan 2023 10:30:00 +0000
                '%Y-%m-%dT%H:%M:%S+00:00'  # 2023-01-25T10:30:00+00:00
            ]
            
            # Clean up date text
            date_text = date_text.strip()
            date_text = re.sub(r'\s+', ' ', date_text)
            
            for date_format in date_formats:
                try:
                    return datetime.strptime(date_text, date_format)
                except ValueError:
                    continue
            
            # Try to extract date using regex for partial match
            date_patterns = [
                r'(\d{2}-\d{2}-\d{4})',  # 01-25-2023
                r'(\d{4}-\d{2}-\d{2})',  # 2023-01-25
                r'(\w+ \d{1,2}, \d{4})',  # January 25, 2023
                r'(\d{1,2} \w+ \d{4})'   # 25 January 2023
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, date_text)
                if match:
                    extracted_date = match.group(1)
                    for date_format in ['%m-%d-%Y', '%Y-%m-%d', '%B %d, %Y', '%d %B %Y']:
                        try:
                            return datetime.strptime(extracted_date, date_format)
                        except ValueError:
                            continue
                            
        except Exception as e:
            logger.warning(f"Error parsing date '{date_text}': {str(e)}")
            
        # Return current date if parsing fails
        return datetime.now(timezone.utc)
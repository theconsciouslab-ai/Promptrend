# collectors/phpbb_collector.py
import logging
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from typing import List, Dict, Any
import asyncio

from agents.forum_agent.collectors.base_collector import BaseForumCollector
from agents.forum_agent.forum_config import(
    INTER_PAGE_DELAY,
)

logger = logging.getLogger("PhpBBCollector")

class PhpBBCollector(BaseForumCollector):
    async def collect_threads(self, forum_url: str, lexicon: List[str], since_time: datetime, language: str) -> List[Dict[str, Any]]:
        threads = []
        logger.info(f"Collecting threads from phpBB forum: {forum_url}")
        try:
            html = await self._get_with_retry(forum_url)
            if not html:
                return []

            soup = BeautifulSoup(html, "html.parser")
            thread_links = soup.select("a.topictitle")

            for link in thread_links:
                href = link.get("href", "")
                thread_url = forum_url.rstrip("/") + "/" + href.lstrip("/")
                thread_data = await self._extract_thread_structure(thread_url)
                if thread_data.get("date") and thread_data["date"] < since_time:
                    continue
                if await self._check_relevance(thread_data.get("content", ""), thread_data.get("title", ""), lexicon):
                    threads.append(thread_data)
                await asyncio.sleep(INTER_PAGE_DELAY)
        except Exception as e:
            logger.error(f"Error in phpBB collection: {str(e)}")
        return threads

    async def _extract_thread_structure(self, thread_url: str) -> Dict[str, Any]:
        try:
            html = await self._get_with_retry(thread_url)
            if not html:
                return {}

            soup = BeautifulSoup(html, "html.parser")
            title_tag = soup.select_one("h2.topic-title")
            title = title_tag.get_text().strip() if title_tag else "Untitled"

            content_blocks = soup.select(".content")
            content = "\n\n".join([c.get_text().strip() for c in content_blocks])

            # ✅ Date parsing patch
            time_tag = soup.find("time")
            if time_tag and time_tag.get("datetime"):
                try:
                    date = datetime.fromisoformat(time_tag["datetime"])
                except Exception:
                    logger.warning(f"Invalid datetime format in time tag: {time_tag.get('datetime')}")
                    date = datetime.now(timezone.utc)
            else:
                date = datetime.now(timezone.utc)

            return {
                "thread_id": thread_url,
                "url": thread_url,
                "title": title,
                "author": "Unknown",
                "date": date,
                "content": content,
                "posts": []
            }
        except Exception as e:
            logger.error(f"Error parsing phpBB thread: {str(e)}")
            return {}

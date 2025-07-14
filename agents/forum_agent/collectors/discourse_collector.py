# collectors/discourse_collector.py

import logging
import json
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from dateutil.parser import parse as parse_date



from agents.forum_agent.collectors.base_collector import BaseForumCollector
from  agents.forum_agent.forum_config import (
    INTER_PAGE_DELAY
)

logger = logging.getLogger("DiscourseCollector")

class DiscourseCollector(BaseForumCollector):
    async def collect_threads(self, forum_url: str, lexicon: List[str], since_time: datetime, language: str) -> List[Dict[str, Any]]:
        threads = []
        logger.info(f"Collecting threads from Discourse forum: {forum_url}")

        try:
            json_url = forum_url.rstrip("/") + "/latest.json"
            content = await self._get_with_retry(json_url)
            if not content:
                return []

            data = json.loads(content)
            topic_list = data.get("topic_list", {}).get("topics", [])
            
            for topic in topic_list:
                try:
                    thread_url = f"{forum_url.rstrip('/')}/t/{topic['slug']}/{topic['id']}"
                    thread_data = await self._extract_thread_structure(thread_url)

                    # Parse thread date safely
                    created_at_raw = topic.get("created_at")
                    if created_at_raw:
                        try:
                            thread_date = parse_date(created_at_raw)
                            if thread_date.tzinfo is None:
                                thread_date = thread_date.replace(tzinfo=timezone.utc)
                            else:
                                thread_date = thread_date.astimezone(timezone.utc)
                        except Exception as e:
                            logger.warning(f"Failed to parse topic date: {e}")
                            thread_date = datetime.now(timezone.utc)
                    else:
                        thread_date = datetime.now(timezone.utc)


                    thread_data["date"] = thread_date

                    if thread_date < since_time:
                        continue

                    if await self._check_relevance(thread_data.get("content", ""), thread_data.get("title", ""), lexicon):
                        threads.append(thread_data)
                        logger.debug(f"Thread added: {thread_data['title'][:60]}")

                    await asyncio.sleep(INTER_PAGE_DELAY)
                except Exception as inner_e:
                    logger.warning(f"Error processing thread: {inner_e}")

        except Exception as e:
            logger.error(f"Error collecting threads from Discourse forum {forum_url}: {str(e)}")

        return threads

    async def _extract_thread_structure(self, thread_url: str) -> Dict[str, Any]:
        try:
            json_url = thread_url.rstrip("/") + ".json"
            raw = await self._get_with_retry(json_url)
            if not raw:
                return {}

            data = json.loads(raw)
            title = data.get("title", "Untitled")
            posts_data = data.get("post_stream", {}).get("posts", [])

            posts = []
            full_text = []

            for post in posts_data:
                post_body = post.get("cooked", "")
                soup = BeautifulSoup(post_body, "html.parser")
                text = soup.get_text(strip=True)
                full_text.append(text)
                posts.append({
                    "author": post.get("username", "Unknown"),
                    "date": post.get("created_at"),
                    "content": text
                })

            thread = {
                "thread_id": thread_url,
                "url": thread_url,
                "title": title,
                "author": posts[0].get("author", "Unknown") if posts else "Unknown",
                "content": "\n\n".join(full_text),
                "posts": posts,
                "date": None  # populated later
            }
            return thread

        except Exception as e:
            logger.error(f"Error parsing Discourse thread JSON {thread_url}: {str(e)}")
            return {}

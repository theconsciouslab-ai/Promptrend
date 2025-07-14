import logging
import asyncio
import os
import json
import re

from agents.forum_agent.forum_agent import ForumAgent as CoreAgent
from processors.metadata_extractor import MetadataExtractor
from processors.llm_analyzer import LLMAnalyzer
from agents.forum_agent.forum_config import FORUM_DATA_PATH

logger = logging.getLogger("DiscussionForumsWrapper")

def sanitize_filename(s):
    return re.sub(r'[^a-zA-Z0-9_-]', '_', s)

class ForumAgent:
    def __init__(self, region_id=None):
        self.region_id = region_id
        self.agent = CoreAgent()
        self.metadata_extractor = MetadataExtractor()
        self.llm_analyzer = LLMAnalyzer()
        self.data_path = FORUM_DATA_PATH

        os.makedirs(self.data_path, exist_ok=True)

    def collect(self):
        logger.info("📚 Loading collected discussion forum threads")
        collected = []

        if not os.path.exists(self.data_path):
            logger.warning(f"Data directory {self.data_path} does not exist.")
            return collected

        for filename in os.listdir(self.data_path):
            if filename.endswith(".json"):
                full_path = os.path.join(self.data_path, filename)
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        item = json.load(f)
                        if isinstance(item, list):
                            collected.extend(item)
                        else:
                            collected.append(item)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load file {filename}: {e}")
        return collected

    def enrich(self, item):
        try:
            # Use clean fallback: URL → thread_id → unknown
            thread_url = (
                item.get("url") or
                item.get("thread_url") or
                item.get("metadata", {}).get("thread_url") or
                item.get("id") or
                "unknown"
            )

            logger.info(f"✨ Enriching forum item {thread_url}")

            content = (
                item.get("content", {}).get("body") or
                item.get("translated_content") or
                item.get("content_snippet") or
                item.get("text") or
                item.get("content") or
                ""
            )

            if not content.strip():
                logger.warning("⚠️ No usable content body found, skipping item")
                return None

            score = asyncio.run(self.llm_analyzer.analyze(content, mode="discussion"))
            metadata = self.metadata_extractor.extract_forum_metadata(item)

            # Patch in identifier fields for cross-agent compatibility
            enriched = {
                **item,
                "platform": "forums",
                "post_id": thread_url,  # so StorageManager picks it up
                "community": {
                    "name": metadata.get("forum_name", "general")  # for StorageManager fallback
                },
                "relevance_score": score,
                "llm_analysis": {"relevance_score": score},
                "metadata": metadata,
                "content": {
                    "body": content
                }
            }

            logger.info(f"✅ Enriched forum item with score {score:.2f}")
            return enriched

        except Exception as e:
            logger.exception(f"❌ Failed to enrich forum item: {e}")
            return None

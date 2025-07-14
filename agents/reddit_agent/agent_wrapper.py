import logging
import asyncio
import os
import json
from agents.reddit_agent.json_storage import JSONStorage
from agents.reddit_agent.reddit_agent import RedditAgent as InternalRedditAgent
from agents.reddit_agent.reddit_config import (
    DATA_DIR,
)
from processors.metadata_extractor import MetadataExtractor
from processors.llm_analyzer import LLMAnalyzer

logger = logging.getLogger("RedditAgent")

class RedditAgent:
    def __init__(self, region_id):
        self.region_id = region_id
        self.storage = JSONStorage(data_dir=DATA_DIR)
        self.metadata_extractor = MetadataExtractor()
        self.llm_analyzer = LLMAnalyzer()
        self.agent = InternalRedditAgent(region_id=region_id)

    def collect(self):
        logger.info("📦 Fetching stored Reddit vulnerabilities")
        results = []
        for filename in os.listdir(self.storage.vulnerabilities_dir):
            if filename.endswith(".json"):
                with open(os.path.join(self.storage.vulnerabilities_dir, filename), "r", encoding="utf-8") as f:
                    item = json.load(f)
                    results.append(item)
        return results


    def enrich(self, item):
        """
        Enrich a single Reddit post using shared metadata and LLM analysis.
        This uses PrompTrend's shared processors.
        """
        try:
            logger.info(f" Enriching Reddit item {item.get('id')}")

            # Extract metadata
            metadata = self.metadata_extractor.extract_reddit_metadata(item)

            # Prepare content for LLM
            content_text = self._prepare_content(item)
            score = asyncio.run(self.llm_analyzer.analyze(content_text, mode="discussion"))

            enriched = {
                **item,
                "platform": "reddit",
                "post_id": item.get("id", "unknown_id"),
                "content": {
                    "body": content_text
                },
                "metadata": metadata,
                "llm_analysis": {
                    "relevance_score": score
                },
                "relevance_score": score
            }

            # Ensure a fallback URL exists for technical indicators
            if "technical_indicators" in enriched["metadata"]:
                enriched["metadata"]["technical_indicators"]["source_url"] = item.get("url", "")

            logger.info(f" Enriched Reddit item {item.get('id')} with score {score:.2f}")
            return enriched

        except Exception as e:
            logger.error(f" Failed to enrich Reddit item: {str(e)}")
            return None


    def _prepare_content(self, item):
        """
        Prepares a Reddit item's title, text, and top comments for LLM scoring.
        """
        content = f"Title: {item.get('title', '')}\n\n"
        content += f"Body: {item.get('selftext', '')}\n\n"
        content += "Top Comments:\n"

        comments = item.get("comments", {}).values() if isinstance(item.get("comments"), dict) else item.get("comments", [])
        top_comments = sorted(comments, key=lambda x: x.get("score", 0), reverse=True)[:5]

        for i, comment in enumerate(top_comments):
            body = comment.get("body", "")
            score = comment.get("score", 0)
            content += f"{i+1}. ({score} pts): {body}\n\n"

        return content.strip()

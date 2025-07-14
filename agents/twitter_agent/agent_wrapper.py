import logging
import asyncio
from agents.twitter_agent.twitter_agent import TwitterAgent as CoreAgent
from processors.metadata_extractor import MetadataExtractor
from processors.llm_analyzer import LLMAnalyzer
from agents.twitter_agent.json_storage import JSONStorage

logger = logging.getLogger("TwitterAgent")

class TwitterAgent:
    def __init__(self, region_id):
        self.region_id = region_id
        self.agent = CoreAgent(region_id=region_id)
        self.storage = JSONStorage()
        self.metadata_extractor = MetadataExtractor()
        self.llm_analyzer = LLMAnalyzer()
        

    def collect(self):
        logger.info("🕊️ Fetching collected Twitter posts")
        return self.storage.get_relevant_tweets(min_score=0.1, limit=50)
    
    def enrich(self, item):
        logger.info(f"⚙️ Enriching Twitter item {item.get('id')}")
        
        content = item.get("text") or item.get("content", "")
        if not content:
            logger.warning(f"❗ Twitter item {item.get('id')} has no usable content")
            return None  # Skip invalid content

        # Score the tweet
        score = asyncio.run(self.llm_analyzer.analyze(content, mode="discussion"))

        # Add metadata
        metadata = self.metadata_extractor.extract_twitter_metadata(item)

        # ✅ Ensure fallback URL is set
        if "technical_indicators" in metadata:
            metadata["technical_indicators"]["source_url"] = item.get("url", "")

        # Inject fields
        item["relevance_score"] = score
        item["llm_analysis"] = {"relevance_score": score}
        item["metadata"] = metadata
        item["platform"] = "twitter"
        item["post_id"] = item.get("id", f"entry_{hash(content)}")

        # ✅ Ensure content.body is present
        item["content"] = {
            "body": content
        }

        return item

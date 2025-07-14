# agents/github_agent/agent_wrapper.py

import logging
import asyncio
import os
import json
from agents.github_agent.github_agent import GithubAgent as CoreAgent
from processors.metadata_extractor import MetadataExtractor
from processors.llm_analyzer import LLMAnalyzer
from agents.github_agent.utils.storage_manager import StorageManager

logger = logging.getLogger("GithubAgentWrapper")

class GithubAgent:
    def __init__(self, region_id=None):
        self.region_id = region_id
        self.agent = CoreAgent(region_id=region_id)
        self.storage = StorageManager()
        self.metadata_extractor = MetadataExtractor()
        self.llm_analyzer = LLMAnalyzer()

    def collect(self):
        logger.info("📦 Fetching stored GitHub vulnerabilities")
        if not self.storage.vuln_file or not os.path.exists(self.storage.vuln_file):
            return []
        
        with open(self.storage.vuln_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Return only items the core agent already flagged as vulnerabilities
        return [item for item in data if item.get("is_vulnerability")]
    
    def enrich(self, item):
        logger.info(f"🔍 Enriching GitHub item {item.get('file_path') or item.get('issue_number')}")
        content = item.get("content_snippet", "")

        # 1) Try to reuse the agent’s existing final_score
        score = item.get("final_score")
        if score is None:
            mode = "code" if item.get("file_path") else "discussion"
            score = asyncio.run(self.llm_analyzer.analyze(content, mode=mode))

        # Add LLM analysis and score
        item["relevance_score"] = score
        item["llm_analysis"] = {"relevance_score": score}

        # Extract and inject metadata
        metadata = self.metadata_extractor.extract_github_metadata(item)
        item["metadata"] = metadata

        # ✅ Add required top-level fields
        item["platform"] = "github"
        item["post_id"] = (
            item.get("file_path")
            or f"issue_{item.get('issue_number')}"
            or f"entry_{hash(content)}"
        )

        # ✅ Ensure content.body exists
        item["content"] = {
            "body": content
        }

        # ✅ Ensure fallback source URL exists
        if "technical_indicators" in item["metadata"]:
            item["metadata"]["technical_indicators"]["source_url"] = item.get("file_url", "")

        return item


    def save_individual_file(self, item):
        file_id = item.get("file_path") or item.get("issue_number") or "unknown"
        safe_name = str(file_id).replace("/", "_").replace(" ", "_")
        path = os.path.join(self.storage.storage_dir, "vulnerabilities", f"{safe_name}.json")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(item, f, indent=2, default=str)

        logger.info(f"✅ Stored individual vulnerability file: {path}")

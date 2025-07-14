import logging
import os
import json
from datetime import datetime
import time
import asyncio
from agents.discord_agent.discord_agent import DiscordAgent as CoreAgent
from processors.metadata_extractor import MetadataExtractor
from agents.discord_agent.processors.llm_analyzer import LLMAnalyzer
from agents.discord_agent.utils.storage_manager import StorageManager

logger = logging.getLogger("DiscordAgentWrapper")

class DiscordAgent:
    """
    Enhanced Discord Agent wrapper focused on LLM-based vulnerability detection.
    
    Removed prompt extraction dependency and enhanced with comprehensive
    LLM analysis for better vulnerability identification.
    """
    
    def __init__(self, region_id=None):
        self.region_id = region_id
        self.agent = CoreAgent()
        self.storage = StorageManager()
        self.metadata_extractor = MetadataExtractor()
        self.llm_analyzer = LLMAnalyzer()

    def collect(self):
        """Collect stored Discord vulnerabilities from individual files."""
        logger.info("📦 Fetching stored Discord vulnerabilities")

        vulnerabilities_dir = os.path.join(self.storage.storage_dir, "vulnerabilities")
        if not os.path.exists(vulnerabilities_dir):
            logger.warning("⚠️ No Discord vulnerabilities directory found.")
            return []

        collected = []
        for filename in os.listdir(vulnerabilities_dir):
            if not filename.endswith(".json"):
                continue
            path = os.path.join(vulnerabilities_dir, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                    # Check multiple possible score fields for compatibility
                    score = (data.get("relevance_score") or 
                            data.get("final_score") or 
                            data.get("overall_score") or 0.0)
                    
                    if data.get("is_vulnerability") and score >= 0.3:
                        collected.append(data)
                        logger.debug(f"✅ Loaded vulnerability: {filename} (score={score:.3f})")
                    else:
                        logger.debug(f"⚠️ Skipped {filename} (score={score:.3f})")
            except Exception as e:
                logger.error(f"❌ Failed to load {filename}: {e}")

        logger.info(f"✅ Loaded {len(collected)} Discord vulnerabilities from individual files")
        return collected

    async def enrich(self, item):
        """
        Enhanced enrichment using comprehensive LLM analysis.
        
        Args:
            item: Raw vulnerability item from collection
            
        Returns:
            dict: Enriched vulnerability data with comprehensive analysis
        """
        logger.info(f"🔍 Enriching Discord item from channel: {item.get('channel_name')}")

        # Extract content from various sources
        content = item.get("content") or ""
        artifacts = item.get("artifacts", {})
        
        # Include text files in content if available
        if not content and "text_files" in artifacts:
            content = "\n".join(artifacts["text_files"])
        
        if not content:
            logger.warning(f"⚠️ No usable content in Discord item {item.get('message_ids')}")
            return None

        # Perform comprehensive LLM analysis
        logger.info("🤖 Performing comprehensive LLM analysis for enrichment...")
        try:
            comprehensive_analysis = await self.llm_analyzer.analyze_content_comprehensive(
                content, artifacts
            )
            
            overall_score = comprehensive_analysis.get("overall_score", 0.0)
            logger.info(f"📊 Enrichment analysis score: {overall_score:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Error during LLM analysis: {e}")
            # Fallback to basic scoring
            overall_score = await self.llm_analyzer.analyze_discussion(content)
            comprehensive_analysis = {
                "overall_score": overall_score,
                "vulnerability_detected": overall_score > 0.5,
                "vulnerability_type": "Unknown",
                "sophistication_level": "unknown",
                "potential_impact": "unknown",
                "key_techniques": [],
                "target_models": [],
                "effectiveness_assessment": "Could not assess",
                "novelty_score": 0.0,
                "confidence": 0.0,
                "potential_mitigations": [],
                "extracted_prompts": [],
                "summary": "Fallback analysis due to processing error"
            }

        # Extract metadata
        metadata = self.metadata_extractor.extract_discord_metadata(item)
        metadata["technical_indicators"]["source_url"] = item.get("message_url", "")

        # Build enriched vulnerability data
        enriched = {
            "id": f"discord__{item.get('server_name')}__{item.get('channel_name')}__{item['message_ids'][0]}".replace(" ", "_"),
            "platform": "discord",
            "server_name": item.get("server_name"),
            "channel_name": item.get("channel_name"),
            "message_ids": item.get("message_ids"),
            "message_url": item.get("message_url"),
            "authors": item.get("authors"),
            "content": {"body": content},
            "artifacts": artifacts,
            "created_at": item.get("timestamp"),
            "collected_at": int(datetime.utcnow().timestamp()),
            
            # Enhanced scoring and analysis
            "relevance_score": overall_score,
            "final_score": overall_score,
            "is_vulnerability": comprehensive_analysis.get("vulnerability_detected", False),
            
            # Comprehensive analysis results
            "comprehensive_analysis": comprehensive_analysis,
            "vulnerability_type": comprehensive_analysis.get("vulnerability_type", "Unknown"),
            "sophistication_level": comprehensive_analysis.get("sophistication_level", "unknown"),
            "potential_impact": comprehensive_analysis.get("potential_impact", "unknown"),
            "key_techniques": comprehensive_analysis.get("key_techniques", []),
            "target_models": comprehensive_analysis.get("target_models", []),
            "extracted_prompts": comprehensive_analysis.get("extracted_prompts", []),
            "potential_mitigations": comprehensive_analysis.get("potential_mitigations", []),
            "analysis_summary": comprehensive_analysis.get("summary", ""),
            
            # Legacy analysis structure for compatibility
            "analysis": {
                "scores": {
                    "discussion": item.get("scores", {}).get("discussion", 0.0),
                    "code": item.get("scores", {}).get("code", 0.0),
                    "overall": overall_score,
                    "confidence": comprehensive_analysis.get("confidence", 0.0)
                },
                "insights": {
                    "vulnerability_type": comprehensive_analysis.get("vulnerability_type", "Unknown"),
                    "target_models": comprehensive_analysis.get("target_models", []),
                    "effectiveness": comprehensive_analysis.get("effectiveness_assessment", "Unknown"),
                    "novelty": f"{comprehensive_analysis.get('novelty_score', 0.0):.2f}",
                    "key_techniques": comprehensive_analysis.get("key_techniques", []),
                    "potential_mitigations": comprehensive_analysis.get("potential_mitigations", [])
                },
                "timestamp": time.time()
            },
            "metadata": metadata
        }

        logger.info(f"✅ Enriched vulnerability with comprehensive analysis")
        logger.info(f"   Type: {enriched['vulnerability_type']}")
        logger.info(f"   Score: {overall_score:.3f}")
        logger.info(f"   Techniques: {len(enriched['key_techniques'])} identified")
        logger.info(f"   Prompts: {len(enriched['extracted_prompts'])} extracted")

        return enriched

    def save_individual_file(self, item):
        """Save enriched vulnerability as individual JSON file."""
        # Build meaningful filename: server__channel__messageid
        server = item.get("server_name") or item.get("platform_data", {}).get("server_name", "unknown_server")
        channel = item.get("channel_name") or item.get("platform_data", {}).get("channel_name", "unknown_channel")
        message_id = item.get("message_ids", ["unknown_msg"])[0]

        safe_name = f"{server}__{channel}__{message_id}".replace(" ", "_").replace("/", "_")
        path = os.path.join(self.storage.storage_dir, "vulnerabilities", f"{safe_name}.json")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Add save timestamp
        item["saved_at"] = datetime.utcnow().isoformat()
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(item, f, indent=2, default=str)

        logger.info(f"✅ Stored enhanced vulnerability file: {path}")
        
        # Log summary of what was stored
        if "comprehensive_analysis" in item:
            analysis = item["comprehensive_analysis"]
            logger.info(f"   Stored analysis: {analysis.get('vulnerability_type', 'Unknown')} "
                       f"(score: {analysis.get('overall_score', 0.0):.3f})")
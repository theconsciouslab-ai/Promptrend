# agents/master_controller.py
import logging
from datetime import datetime
import asyncio


from agents.regional_coordinator import RegionalCoordinator
from database.storage_manager import StorageManager
from processors.transformation_pipeline import TransformationPipeline
from processors.pvaf.pvaf_calculator import PVAFCalculator
from coordinator.cross_platform_coordinator import CrossPlatformCoordinator

import data_collection_config



def ensure_iso_timestamp(val):
    if isinstance(val, datetime):
        return val.isoformat()
    try:
        return datetime.fromisoformat(val).isoformat()
    except Exception:
        return datetime.utcnow().isoformat()


logger = logging.getLogger("MasterController")


class MasterController:
    """
    Central coordinator for the PrompTrend data collection system.

    Manages the three-stage pipeline:
    1. Raw Data Collection: Platform agents collect raw content
    2. Metadata Enrichment: Raw content is enhanced with metadata
    3. Database Integration: Processed data is stored in the database
    """

    def __init__(self):
        """Initialize the master controller."""
        self.regional_coordinators = []
        self.storage_manager = StorageManager()
        self.transformation_pipeline = TransformationPipeline(self.storage_manager)
        self.cross_platform_coordinator = CrossPlatformCoordinator()
        self.pvaf_calculator = PVAFCalculator(use_llm_judge=True, scoring_mode="collection")
        self.collection_stats = {
            "start_time": None,
            "end_time": None,
            "raw_items_collected": 0,
            "processed_items": 0,
            "vulnerabilities_identified": 0,
        }

    def initialize_agents(self):
        """Initialize regional coordinators and platform agents."""
        logger.info("Initializing agent hierarchy")

        # Create regional coordinators based on configuration
        for region_config in data_collection_config.REGIONAL_CONFIGS:
            coordinator = RegionalCoordinator(
                region_id=region_config["id"], platforms=region_config["platforms"]
            )
            coordinator.initialize_platform_agents()
            self.regional_coordinators.append(coordinator)

        logger.info(
            f"Initialized {len(self.regional_coordinators)} regional coordinators"
        )

    def run_collection_pipeline(self):
        """Execute the three-stage data collection pipeline."""
        self.collection_stats["start_time"] = datetime.now()

        try:
            # Stage 1: Raw Data Collection
            raw_data = self._execute_raw_collection()

            # Stage 2: Metadata Enrichment
            enriched_data = self._execute_metadata_enrichment(raw_data)

            # Stage 3: Database Integration
            self._execute_database_integration(enriched_data)

            self.collection_stats["end_time"] = datetime.now()
            duration = (
                self.collection_stats["end_time"] - self.collection_stats["start_time"]
            ).total_seconds()

            logger.info(f"Pipeline completed in {duration:.2f} seconds")
            logger.info(f"Stats: {self.collection_stats}")

        except Exception as e:
            logger.error(f"Error in collection pipeline: {str(e)}")
            self.collection_stats["end_time"] = datetime.now()

    def _execute_raw_collection(self):
        """Execute the raw data collection stage."""
        logger.info("Starting raw data collection stage")

        raw_items = []
        for coordinator in self.regional_coordinators:
            items = coordinator.collect_raw_data()
            raw_items.extend(items)

        self.collection_stats["raw_items_collected"] = len(raw_items)
        logger.info(f"Raw collection completed: {len(raw_items)} items")

        return raw_items

    def _execute_metadata_enrichment(self, raw_items):
        """
        Execute the metadata enrichment stage.

        Args:
            raw_items: List of raw collected items

        Returns:
            list: Enriched items with metadata
        """
        logger.info("Starting metadata enrichment stage")

        enriched_items = []
        for item in raw_items:
            # Process each item through the enrichment pipeline
            enriched_item = self._enrich_item(item)
            if enriched_item:
                enriched_items.append(enriched_item)

        self.collection_stats["processed_items"] = len(enriched_items)
        logger.info(f"Metadata enrichment completed: {len(enriched_items)} items")

        return enriched_items


    def _enrich_item(self, item):
        region_id = item.get("collected_by_region")
        
        # 🔍 DEBUG: Check what Reddit agent produced
        logger.info(f"🔍 [DEBUG] ===== BEFORE ENRICHMENT =====")
        logger.info(f"🔍 [DEBUG] Item ID: {item.get('id', 'unknown')}")
        logger.info(f"🔍 [DEBUG] Platform: {item.get('platform', 'unknown')}")
        logger.info(f"🔍 [DEBUG] Item keys: {list(item.keys())}")
        
        # Check for metadata in the item as received from Reddit agent
        if "metadata" in item:
            metadata = item["metadata"]
            logger.info(f"🔍 [DEBUG] Found metadata with keys: {list(metadata.keys())}")
            
            if "social_signals" in metadata:
                social = metadata["social_signals"]
                logger.info(f"🔍 [DEBUG] Social signals keys: {list(social.keys())}")
                
                if "engagement_metrics" in social:
                    engagement = social["engagement_metrics"]
                    logger.info(f"🔍 [DEBUG] Engagement metrics: {engagement}")
        else:
            logger.info(f"🔍 [DEBUG] No 'metadata' key found in item")
            
            # Check if it's in platform_data instead
            platform_data = item.get("platform_data", {})
            if platform_data:
                logger.info(f"🔍 [DEBUG] Platform data keys: {list(platform_data.keys())}")
        
        for coordinator in self.regional_coordinators:
            if coordinator.region_id == region_id:
                enriched = coordinator.enrich_item(item)

                if asyncio.iscoroutine(enriched):
                    enriched = asyncio.run(enriched)

                if enriched is None:
                    logger.warning(f"Skipped enrichment for item from region {region_id}")
                    return None

                try:
                    # 🔍 DEBUG: Check what coordinator.enrich_item returned
                    logger.info(f"🔍 [DEBUG] ===== AFTER COORDINATOR ENRICHMENT =====")
                    logger.info(f"🔍 [DEBUG] Enriched item keys: {list(enriched.keys())}")
                    
                    if "metadata" in enriched:
                        metadata = enriched["metadata"]
                        logger.info(f"🔍 [DEBUG] Enriched metadata keys: {list(metadata.keys())}")
                        
                        if "social_signals" in metadata:
                            social = metadata["social_signals"]
                            if "engagement_metrics" in social:
                                engagement = social["engagement_metrics"]
                                logger.info(f"🔍 [DEBUG] Final engagement metrics: {engagement}")
                                logger.info(f"🔍 [DEBUG] Final engagement score: {engagement.get('engagement_score', 'NOT FOUND')}")
                    
                    logger.debug(f"Enriched item structure: {enriched}")
                    
                    # Try to extract some usable content field (rest of your existing logic...)
                    possible_fields = [
                        ("content", "body"),
                        ("content_snippet",),
                        ("text",),
                        ("raw_text",),
                        ("summary",),
                        ("prompt",),
                    ]

                    content = None
                    for path in possible_fields:
                        value = enriched
                        for key in path:
                            if isinstance(value, dict):
                                value = value.get(key)
                            else:
                                value = None
                                break
                        if value:
                            content = value
                            break

                    if not content:
                        raise ValueError("Missing 'content.body' or fallback text")

                    platform = enriched.get("platform", "unknown")
                    metadata = enriched

                    if "url" not in metadata:
                        metadata["url"] = (
                            metadata.get("file_url")
                            or metadata.get("repo_url")
                            or metadata.get("post_id", "unknown")
                        )

                    # 🔐 Ensure all timestamps are ISO strings before passing to CrossPlatformCoordinator
                    temporal = metadata.get("metadata", {}).get("temporal_data", {})
                    if "collection_timestamp" in temporal:
                        temporal["collection_timestamp"] = ensure_iso_timestamp(
                            temporal["collection_timestamp"]
                        )
                    if "discovery_timestamp" in temporal:
                        temporal["discovery_timestamp"] = ensure_iso_timestamp(
                            temporal["discovery_timestamp"]
                        )
                    for entry in temporal.get("propagation_timeline", []):
                        if "timestamp" in entry:
                            entry["timestamp"] = ensure_iso_timestamp(
                                entry["timestamp"]
                            )

                    vuln_id, is_new = asyncio.run(
                        self.cross_platform_coordinator.process_new_content(
                            content=content, platform=platform, metadata=metadata
                        )
                    )

                    enriched["vulnerability_id"] = vuln_id
                    lifecycle = (
                        self.cross_platform_coordinator.get_vulnerability_lifecycle(
                            vuln_id
                        )
                    )
                    enriched["vulnerability_stage"] = lifecycle.get("evolution_stage")

                    # 🧠 PVAF scoring
                    try:
                        pvaf_result = asyncio.run(self.pvaf_calculator.calculate_pvaf(enriched))

                        enriched["pvaf_score"] = pvaf_result["final_score"]
                        enriched["pvaf_classification"] = pvaf_result["classification"]
                        enriched["pvaf_details"] = {
                            "base_score": pvaf_result["base_score"],
                            "modifiers": pvaf_result["modifiers"],
                            "dimension_scores": pvaf_result["dimension_scores"],
                            "dimension_evidence": pvaf_result["dimension_evidence"],
                            "timestamp": pvaf_result["timestamp"],
                        }

                    except Exception as e:
                        logger.warning(
                            f"PVAF scoring failed for item {enriched.get('id', 'unknown')}: {str(e)}"
                        )

                except Exception as e:
                    logger.warning(f"CrossPlatformCoordinator error: {e}")
                    
                logger.info(f"Item {enriched.get('id', 'N/A')} scored {enriched.get('pvaf_score', '?')} ({enriched.get('pvaf_classification', 'unknown')})")
                return enriched
                
        logger.warning(f"No coordinator found for region {region_id}")
        return None

    def _execute_database_integration(self, enriched_items):
        """
        Execute the database integration stage.

        Args:
            enriched_items: List of enriched items
        """
        logger.info("Starting database integration stage")

        stored_count = 0  

        for item in enriched_items:
            if self._is_vulnerability(item):
                platform = item.get("platform", "unknown")

                # Diagnostic logging before transformation
                if not item.get("cleaned_prompts"):
                    logger.warning(f"Item {item.get('id')} has no cleaned prompts")

                elif len(item["cleaned_prompts"][0].get("cleaned_prompt", "").strip()) < 30:
                    logger.warning(f"Item {item.get('id')} has a short cleaned prompt")

                transformed = self.transformation_pipeline.process_item(item, platform=platform)
                if transformed:
                    was_stored = self.storage_manager.store_vulnerability(transformed)
                    if was_stored:
                        stored_count += 1

                self.collection_stats["vulnerabilities_identified"] += 1

        logger.info(
            f"Database integration completed: {stored_count} new vulnerabilities stored out of {self.collection_stats['vulnerabilities_identified']} total identified"
        )


       
    def _is_vulnerability(self, item):
        """
        Determine if an enriched item represents a vulnerability.

        Args:
            item: Enriched item

        Returns:
            bool: True if the item is a vulnerability
        """
        # Check if the item meets the vulnerability threshold
        relevance_score = item.get("relevance_score", 0.0)
        return relevance_score >= data_collection_config.VULNERABILITY_THRESHOLD

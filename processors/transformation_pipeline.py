# processors/transformation_pipeline.py
import logging
from datetime import datetime
import asyncio

from processors.llm_prompt_cleaner import LLMPromptCleaner
import data_collection_config

logger = logging.getLogger("TransformationPipeline")

class TransformationPipeline:
    """
    Processes data through multiple transformation stages.
    
    Implements a four-stage pipeline:
    1. Raw Ingestion: Preserves unmodified content with source metadata
    2. Normalization: Standardizes format and applies privacy transformations
    3. Enrichment: Augments with derived attributes and cross-references
    4. Analytical: Generates aggregations and time-series projections
    """
    
    def __init__(self, storage_manager):
        """
        Initialize the transformation pipeline.
        
        Args:
            storage_manager: Storage manager for database operations
        """
        self.storage_manager = storage_manager
        self.transformation_history = {}
        self.prompt_cleaner = LLMPromptCleaner()
        
    def process_item(self, item, platform):
        """
        Process an item through the transformation pipeline.
        
        Args:
            item: Raw collected item
            platform: Source platform
            
        Returns:
            dict: Processed item
        """
        item_id = item.get("id")
        
        # Record transformation start
        self._record_transformation_start(item_id, platform)
        
        try:
            # Stage 1: Raw Ingestion
            raw_item = self._raw_ingestion_stage(item, platform)
            
            # Stage 2: Normalization
            normalized_item = self._normalization_stage(raw_item, platform)
            
            # Stage 3: Enrichment
            enriched_item = self._enrichment_stage(normalized_item, platform)
            
            # Stage 4: Analytical
            analytical_item = self._analytical_stage(enriched_item, platform)
            
            # Record transformation completion
            self._record_transformation_completion(item_id, platform)
            logger.info(f"✅ Transformed item ID: {analytical_item.get('id')}")

            return analytical_item
        
            
        except Exception as e:
            logger.error(f"Error in transformation pipeline for {platform} item {item_id}: {str(e)}")
            self._record_transformation_error(item_id, platform, str(e))
            return None
    
    def _raw_ingestion_stage(self, item, platform):
        """
        Process item through raw ingestion stage.
        
        Args:
            item: Original item
            platform: Source platform
            
        Returns:
            dict: Item with raw ingestion metadata
        """
        # Add raw ingestion metadata
        raw_item = {**item}
        raw_item["_raw_metadata"] = {
            "pipeline_ingestion_time": datetime.now().isoformat(),
            "source_platform": platform,
            "raw_schema_version": data_collection_config.SCHEMA_VERSIONS["raw"]
        }
        
        # Record stage completion
        self._record_transformation_stage(item.get("id"), "raw_ingestion")
        
        return raw_item
    
    def _normalization_stage(self, item, platform):
        """
        Process item through normalization stage.
        
        Args:
            item: Raw ingested item
            platform: Source platform
            
        Returns:
            dict: Normalized item
        """
        # Apply platform-specific normalization
        normalizer_method = f"_normalize_{platform}_item"
        
        if hasattr(self, normalizer_method):
            normalized_item = getattr(self, normalizer_method)(item)
        else:
            # Default normalization
            normalized_item = self._normalize_generic_item(item)
        
        # Add normalization metadata
        normalized_item["_norm_metadata"] = {
            "normalization_time": datetime.now().isoformat(),
            "normalized_schema_version": data_collection_config.SCHEMA_VERSIONS["normalized"]
        }
        
        # Record stage completion
        self._record_transformation_stage(item.get("id"), "normalization")
        
        return normalized_item
    
    def _normalize_generic_item(self, item):
        """
        Apply generic normalization to an item.

        Args:
            item: Raw ingested item

        Returns:
            dict: Normalized item
        """
        normalized = {}

        # Copy standard fields
        normalized["id"] = (
            item.get("id")
            or item.get("file_path")
            or item.get("filename")
            or item.get("url")
            or item.get("title")
            or "unknown"
        )

        normalized["platform"] = item.get("_raw_metadata", {}).get("source_platform")
        normalized["collection_time"] = item.get("_raw_metadata", {}).get("pipeline_ingestion_time")
        normalized["type"] = item.get("type")

        # Content section
        normalized["content"] = {}

        if "title" in item:
            normalized["content"]["title"] = item["title"]
        elif "subject" in item:
            normalized["content"]["title"] = item["subject"]

        # Content body fallback logic
        if "text" in item:
            normalized["content"]["body"] = item["text"]
        elif "body" in item:
            normalized["content"]["body"] = item["body"]
        elif "selftext" in item:
            normalized["content"]["body"] = item["selftext"]
        elif "selftext" in item.get("platform_data", {}):
            normalized["content"]["body"] = item["platform_data"]["selftext"]
        elif "content" in item:
            if isinstance(item["content"], str):
                normalized["content"]["body"] = item["content"]
            elif isinstance(item["content"], dict) and "body" in item["content"]:
                normalized["content"]["body"] = item["content"]["body"]
        else:
            normalized["content"]["body"] = ""

        # Author logic
        normalized["author"] = item.get("author")
        if not normalized.get("author") and "authors" in item.get("platform_data", {}):
            authors = item["platform_data"]["authors"]
            if isinstance(authors, list) and authors:
                normalized["author"] = authors[0]

        # URL logic
        if "url" in item:
            normalized["url"] = item["url"]
        elif "permalink" in item:
            normalized["url"] = item["permalink"]

        # Timestamps
        if "created_utc" in item:
            normalized["created_at"] = item["created_utc"]
        elif "timestamp" in item:
            normalized["created_at"] = item["timestamp"]
        elif "created_at" in item:
            normalized["created_at"] = item["created_at"]

        # Interactions
        normalized["interactions"] = {}

        # Platform-specific leftovers
        normalized["platform_data"] = {
            k: v for k, v in item.items()
            if k not in ["id", "type", "title", "text", "body", "content", 
                        "author", "url", "permalink", "created_utc", 
                        "timestamp", "created_at", "_raw_metadata"]
        }

        # Promote important fields from platform_data
        promote_fields = [
            "is_vulnerability", "relevance_score", "final_score", "scores",
            "extracted_prompts", "commit_sha", "repo_name", "file_path", "issue_number", "discussion_id"
        ]
        for field in promote_fields:
            if field in normalized["platform_data"]:
                normalized[field] = normalized["platform_data"][field]

        # Fallback to content_snippet if body is still empty
        if (
            not normalized["content"].get("body")
            and "content_snippet" in normalized["platform_data"]
        ):
            normalized["content"]["body"] = normalized["platform_data"]["content_snippet"]

        # Ensure platform fallback
        if not normalized.get("platform"):
            normalized["platform"] = item.get("_raw_metadata", {}).get("source_platform", "unknown")

        return normalized
    
    def _normalize_reddit_item(self, item):
        """
        Apply Reddit-specific normalization.
        
        Args:
            item: Raw Reddit item
            
        Returns:
            dict: Normalized Reddit item
        """
        # Start with generic normalization
        normalized = self._normalize_generic_item(item)
        
        # Add Reddit-specific interaction fields
        normalized["interactions"] = {
            "upvotes": item.get("score", 0),
            "upvote_ratio": item.get("upvote_ratio", 0.5),
            "comments": item.get("num_comments", 0)
        }
        
        # Extract subreddit information
        normalized["community"] = {
            "name": item.get("subreddit"),
            "type": "subreddit"
        }
        
        # Process comments specifically for Reddit
        if "comments" in item:
            normalized["replies"] = self._normalize_reddit_comments(item["comments"])
        
        return normalized
    
    def _normalize_reddit_comments(self, comments, depth=0):
        """
        Normalize Reddit comments.
        
        Args:
            comments: List of comment objects
            depth: Current depth level
            
        Returns:
            list: Normalized comments
        """
        normalized_comments = []
        
        for comment in comments:
            normalized_comment = {
                "id": comment.get("id"),
                "author": comment.get("author"),
                "content": comment.get("body"),
                "created_at": comment.get("created_utc"),
                "score": comment.get("score", 0),
                "depth": depth,
                "replies": []
            }
            
            # Process replies recursively
            if "replies" in comment and comment["replies"]:
                normalized_comment["replies"] = self._normalize_reddit_comments(
                    comment["replies"], depth + 1
                )
            
            normalized_comments.append(normalized_comment)
        
        return normalized_comments
    
    def _enrichment_stage(self, item, platform):
        """
        Process item through enrichment stage.
        🔧 DEBUGGING: Let's see what's happening with metadata

        Args:
            item: Normalized item
            platform: Source platform

        Returns:
            dict: Enriched item
        """
        # Create a deep copy of the normalized item
        enriched_item = {**item}

        # 🔍 DEBUG: Let's see what we're receiving
        logger.info(f"🔍 [DEBUG] Processing item for platform: {platform}")
        logger.info(f"🔍 [DEBUG] Item keys: {list(item.keys())}")
        
        # Check for metadata in different locations
        direct_metadata = item.get("metadata")
        platform_metadata = item.get("platform_data", {}).get("metadata")
        nested_metadata = item.get("platform_data", {}).get("social_signals")
        
        logger.info(f"🔍 [DEBUG] Direct metadata exists: {direct_metadata is not None}")
        logger.info(f"🔍 [DEBUG] Platform metadata exists: {platform_metadata is not None}")
        logger.info(f"🔍 [DEBUG] Nested social signals exist: {nested_metadata is not None}")
        
        if direct_metadata:
            logger.info(f"🔍 [DEBUG] Direct metadata keys: {list(direct_metadata.keys())}")
            if "social_signals" in direct_metadata:
                logger.info(f"🔍 [DEBUG] Social signals keys: {list(direct_metadata['social_signals'].keys())}")
        
        # ✅ Step 1: Extract candidate prompt texts (existing logic)
        raw_prompts = []

        # (1) From main message body
        body = item.get("content", {}).get("body", "")
        if body and len(body.strip()) > 20:
            raw_prompts.append(body)

        # (2) From artifacts like text files
        artifacts = item.get("platform_data", {}).get("artifacts", {})
        for key, files in artifacts.items():
            if isinstance(files, list):
                for file_content in files:
                    if isinstance(file_content, str) and len(file_content.strip()) > 20:
                        raw_prompts.append(file_content)

        # (3) From platform_data.prompts[]
        platform_prompts = item.get("platform_data", {}).get("prompts", [])
        for p in platform_prompts:
            if isinstance(p, str):
                raw_prompts.append(p)
            elif isinstance(p, dict):
                raw_prompts.append(p.get("text") or p.get("content", ""))

        # ✅ Step 2: Clean prompts using LLM with fallback to raw (existing logic)
        cleaned_prompts = []
        for text in raw_prompts:
            if not text.strip():
                continue

            try:
                cleaned = asyncio.run(self.prompt_cleaner.clean_prompt(text))
                prompt_final = cleaned.strip() if cleaned else text.strip()

                cleaned_prompts.append({
                    "cleaned_prompt": prompt_final,
                    "source": text,
                    "platform": platform
                })

            except Exception as e:
                logger.warning(f"[Cleaner] LLM error, fallback to raw: {e}")
                cleaned_prompts.append({
                    "cleaned_prompt": text.strip(),
                    "source": text,
                    "platform": platform
                })

        enriched_item["cleaned_prompts"] = cleaned_prompts

        # 🆕 Step 3: ENHANCED - Find metadata wherever it exists
        existing_metadata = None
        
        # Try multiple locations for metadata
        if item.get("metadata"):
            existing_metadata = item["metadata"]
            logger.info("🔍 [DEBUG] Found metadata in item['metadata']")
        elif item.get("platform_data", {}).get("metadata"):
            existing_metadata = item["platform_data"]["metadata"]
            logger.info("🔍 [DEBUG] Found metadata in item['platform_data']['metadata']")
        elif item.get("platform_data", {}).get("social_signals"):
            # Create metadata structure from loose social signals
            existing_metadata = {
                "social_signals": item["platform_data"]["social_signals"],
                "temporal_data": item["platform_data"].get("temporal_data", {}),
                "technical_indicators": item["platform_data"].get("technical_indicators", {})
            }
            logger.info("🔍 [DEBUG] Reconstructed metadata from platform_data social_signals")
        else:
            logger.info("🔍 [DEBUG] No metadata found anywhere, will create new")
        
        if existing_metadata:
            # If metadata already exists (from agents), enhance it
            logger.info(f"Found existing metadata from {platform} agent, enhancing it")
            enriched_metadata = self._enhance_existing_metadata(existing_metadata, item, platform)
        else:
            # If no metadata exists, create from scratch
            logger.info(f"No existing metadata found, creating new metadata structure")
            enriched_metadata = self._create_community_metadata(item, platform)
        
        enriched_item["metadata"] = enriched_metadata
        
        # 🔍 DEBUG: Log final engagement score
        final_engagement_score = enriched_metadata.get("social_signals", {}).get("engagement_metrics", {}).get("engagement_score", 0)
        logger.info(f"🔍 [DEBUG] Final engagement score: {final_engagement_score}")
        
        # 🆕 Step 4: Initialize testing history structure for Temporal Resilience
        enriched_item["testing_history"] = self._initialize_testing_history(item)
        
        # 🆕 Step 5: Initialize execution results structure for Cross-Platform Efficacy
        enriched_item["execution_results"] = self._initialize_execution_results(item)

        # ✅ Step 6: Continue normal enrichment (existing logic)
        enriched_item["derived"] = {
            "content_length": self._calculate_content_length(item),
            "reading_time": self._estimate_reading_time(item),
            "language": self._detect_language(item),
            "sentiment": self._analyze_sentiment(item)
        }

        enriched_item["cross_references"] = self._extract_cross_references(item)
        enriched_item["topics"] = self._extract_topics(item)
        enriched_item["entities"] = self._extract_entities(item)

        enriched_item["_enrichment_metadata"] = {
            "enrichment_time": datetime.now().isoformat(),
            "enrichment_schema_version": data_collection_config.SCHEMA_VERSIONS["enriched"]
        }

        self._record_transformation_stage(item.get("id"), "enrichment")

        return enriched_item


    def _enhance_existing_metadata(self, existing_metadata, item, platform):
        """
        🆕 Enhance existing metadata from agents instead of overwriting it
        
        Args:
            existing_metadata: Metadata already created by platform agents
            item: Normalized item
            platform: Source platform
            
        Returns:
            dict: Enhanced metadata
        """
        enhanced = {**existing_metadata}  # Start with existing metadata
        
        # Enhance social signals if they exist
        if "social_signals" in enhanced:
            existing_social = enhanced["social_signals"]
            
            # Enhance engagement metrics
            if "engagement_metrics" in existing_social:
                engagement = existing_social["engagement_metrics"]
                
                # Recalculate engagement score if we have the raw metrics
                if "engagement_score" not in engagement or engagement["engagement_score"] == 0:
                    interactions = {
                        "upvotes": engagement.get("upvotes", 0),
                        "downvotes": engagement.get("downvotes", 0), 
                        "comments": engagement.get("comments", 0),
                        "shares": engagement.get("shares", 0),
                        "likes": engagement.get("likes", 0),
                        "retweets": engagement.get("retweets", 0),
                        "views": engagement.get("views", 1)
                    }
                    
                    # Use our enhanced calculation
                    engagement["engagement_score"] = self._calculate_engagement_score(interactions)
                    engagement["platform"] = platform
            
            # Enhance discussion depth if missing or incomplete
            if "discussion_depth" not in existing_social or existing_social["discussion_depth"]["max_thread_length"] == 0:
                existing_social["discussion_depth"] = {
                    "max_thread_length": self._calculate_max_thread_length(item),
                    "avg_response_depth": self._calculate_avg_response_depth(item),
                    "branches": self._count_discussion_branches(item)
                }
            
            # Enhance community validation if missing
            if "community_validation" not in existing_social:
                platform_data = item.get("platform_data", {})
                existing_social["community_validation"] = {
                    "success_confirmations": platform_data.get("success_confirmations", 0),
                    "failure_reports": platform_data.get("failure_reports", 0),
                    "validation_ratio": self._calculate_validation_ratio(platform_data)
                }
            
            # Enhance cross references if missing
            if "cross_references" not in existing_social:
                existing_social["cross_references"] = {
                    "mentioned_in_discussions": 0,
                    "linked_from_other_vulnerabilities": 0
                }
        
        # Add community info if missing
        if "community_info" not in enhanced:
            enhanced["community_info"] = self._create_community_info(item, platform)
        
        # Enhance temporal data
        if "temporal_data" in enhanced:
            # Ensure propagation timeline exists
            if "propagation_timeline" not in enhanced["temporal_data"]:
                enhanced["temporal_data"]["propagation_timeline"] = [
                    {
                        "platform": platform,
                        "timestamp": item.get("created_at") or datetime.now().isoformat()
                    }
                ]
        else:
            # Create temporal data if missing
            enhanced["temporal_data"] = {
                "collection_timestamp": item.get("collection_time") or datetime.now().isoformat(),
                "discovery_timestamp": enhanced.get("temporal_data", {}).get("discovery_timestamp") or item.get("created_at") or datetime.now().isoformat(),
                "propagation_timeline": enhanced.get("temporal_data", {}).get("propagation_timeline", [
                    {
                        "platform": platform,
                        "timestamp": item.get("created_at") or datetime.now().isoformat()
                    }
                ])
            }
        
        # Ensure platform is set
        enhanced["platform"] = platform
        
        logger.info(f"Enhanced existing metadata for {platform}: engagement_score={enhanced.get('social_signals', {}).get('engagement_metrics', {}).get('engagement_score', 0)}")
        
        return enhanced

    def _create_community_info(self, item, platform):
        """
        Create community info section for metadata
        
        Args:
            item: Normalized item
            platform: Source platform
            
        Returns:
            dict: Community info
        """
        platform_data = item.get("platform_data", {})
        interactions = item.get("interactions", {})
        
        if platform == "reddit":
            return {
                "subreddit": item.get("community", {}).get("name") or platform_data.get("subreddit"),
                "subreddit_size": platform_data.get("subreddit_subscribers", 0),
                "post_flair": platform_data.get("link_flair_text"),
                "is_pinned": platform_data.get("stickied", False),
                "upvote_ratio": interactions.get("upvote_ratio", 0.5)
            }
        elif platform == "github":
            return {
                "repository": platform_data.get("repo_name"),
                "stars": platform_data.get("stargazers_count", 0),
                "forks": platform_data.get("forks_count", 0),
                "watchers": platform_data.get("watchers_count", 0),
                "is_trending": platform_data.get("is_trending", False),
                "language": platform_data.get("language"),
                "topics": platform_data.get("topics", [])
            }
        elif platform == "discord":
            return {
                "server": platform_data.get("server"),
                "channel": platform_data.get("channel"),
                "member_count": platform_data.get("member_count", 0),
                "is_public": platform_data.get("is_public", False),
                "server_boost_level": platform_data.get("boost_level", 0)
            }
        elif platform == "twitter":
            return {
                "followers": platform_data.get("followers_count", 0),
                "retweets": interactions.get("retweets", 0),
                "likes": interactions.get("likes", 0),
                "is_verified": platform_data.get("verified", False),
                "account_age": platform_data.get("account_created_at")
            }
        else:
            return {"platform": platform}

    def _create_community_metadata(self, item, platform):
        """
        🆕 Create properly structured metadata for Community Adoption calculator
        🔧 ENHANCED: Better extraction from item data
        """
        logger.info(f"🔍 [DEBUG] Creating community metadata for {platform}")
        
        # Extract social signals from various sources
        interactions = item.get("interactions", {})
        platform_data = item.get("platform_data", {})
        
        # 🔍 DEBUG: Log what we found
        logger.info(f"🔍 [DEBUG] Interactions found: {interactions}")
        logger.info(f"🔍 [DEBUG] Platform data keys: {list(platform_data.keys()) if platform_data else 'None'}")
        
        # Build social signals structure with better fallbacks
        social_signals = {
            "engagement_metrics": {
                "platform": platform,
                "upvotes": (
                    interactions.get("upvotes", 0) or 
                    platform_data.get("score", 0) or 
                    platform_data.get("ups", 0) or 0
                ),
                "downvotes": (
                    interactions.get("downvotes", 0) or 
                    platform_data.get("downs", 0) or 0
                ),
                "comments": (
                    interactions.get("comments", 0) or 
                    platform_data.get("num_comments", 0) or 
                    platform_data.get("comment_count", 0) or 0
                ),
                "shares": interactions.get("shares", 0),
                "views": interactions.get("views", 0),
                "likes": interactions.get("likes", 0),
                "retweets": interactions.get("retweets", 0)
            }
        }
        
        # Calculate engagement score with the extracted metrics
        social_signals["engagement_metrics"]["engagement_score"] = self._calculate_engagement_score(
            social_signals["engagement_metrics"]
        )
        
        logger.info(f"🔍 [DEBUG] Calculated engagement score: {social_signals['engagement_metrics']['engagement_score']}")
        
        # Discussion depth analysis
        social_signals["discussion_depth"] = {
            "max_thread_length": self._calculate_max_thread_length(item),
            "avg_response_depth": self._calculate_avg_response_depth(item),
            "branches": self._count_discussion_branches(item)
        }
        
        # Community validation
        social_signals["community_validation"] = {
            "success_confirmations": platform_data.get("success_confirmations", 0),
            "failure_reports": platform_data.get("failure_reports", 0),
            "validation_ratio": self._calculate_validation_ratio(platform_data)
        }
        
        # Cross references
        social_signals["cross_references"] = {
            "mentioned_in_discussions": platform_data.get("mentions", 0),
            "linked_from_other_vulnerabilities": 0  # Will be updated by relationship manager
        }
        
        # Platform-specific community info
        community_info = self._create_community_info(item, platform)
        
        # Build temporal data
        temporal_data = {
            "collection_timestamp": item.get("collection_time") or datetime.now().isoformat(),
            "discovery_timestamp": (
                item.get("created_at") or 
                item.get("created_utc") or 
                platform_data.get("created_utc") or 
                datetime.now().isoformat()
            ),
            "propagation_timeline": [
                {
                    "platform": platform,
                    "timestamp": (
                        item.get("created_at") or 
                        item.get("created_utc") or 
                        platform_data.get("created_utc") or 
                        datetime.now().isoformat()
                    )
                }
            ]
        }
        
        metadata = {
            "social_signals": social_signals,
            "community_info": community_info,
            "temporal_data": temporal_data,
            "platform": platform
        }
        
        logger.info(f"🔍 [DEBUG] Created metadata with engagement score: {metadata['social_signals']['engagement_metrics']['engagement_score']}")
        
        return metadata

    # Add this debugging method to see what's in your Reddit agent data
    def debug_item_structure(self, item, stage_name):
        """
        🔍 DEBUG: Print item structure at different stages
        """
        logger.info(f"🔍 [DEBUG] ===== {stage_name} =====")
        logger.info(f"🔍 [DEBUG] Item ID: {item.get('id', 'unknown')}")
        logger.info(f"🔍 [DEBUG] Platform: {item.get('platform', 'unknown')}")
        
        # Check for metadata locations
        locations_to_check = [
            ("item['metadata']", item.get("metadata")),
            ("item['platform_data']", item.get("platform_data")),
            ("item['interactions']", item.get("interactions")),
            ("item['social_signals']", item.get("social_signals")),
        ]
        
        for location_name, data in locations_to_check:
            if data:
                logger.info(f"🔍 [DEBUG] {location_name}: {type(data)} with keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                
                # If it's social signals, show the details
                if "social" in location_name.lower() and isinstance(data, dict):
                    if "engagement_metrics" in data:
                        metrics = data["engagement_metrics"]
                        logger.info(f"🔍 [DEBUG]   Engagement metrics: {metrics}")
            else:
                logger.info(f"🔍 [DEBUG] {location_name}: None/Empty")

    def _initialize_testing_history(self, item):
        """
        🆕 Initialize testing history structure for Temporal Resilience calculator
        """
        return {
            "tests_conducted": [],
            "resilience_over_time": [],
            "adaptation_attempts": [],
            "mitigation_effectiveness": {},
            "temporal_analysis": {
                "first_test_date": None,
                "last_test_date": None,
                "test_frequency": 0,
                "success_rate_over_time": []
            },
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "version": "1.0"
            }
        }

    def _initialize_execution_results(self, item):
        """
        🆕 Initialize execution results structure for Cross-Platform Efficacy calculator
        """
        return {
            "model_results": {},  # Will be populated during benchmark phase
            "platform_analysis": {},
            "cross_model_comparison": {},
            "effectiveness_metrics": {
                "overall_success_rate": 0.0,
                "model_specific_rates": {},
                "transformation_effectiveness": {}
            },
            "statistical_analysis": {
                "confidence_intervals": {},
                "significance_tests": {},
                "effect_sizes": {}
            },
            "metadata": {
                "total_tests": 0,
                "models_tested": [],
                "transformations_tested": [],
                "last_updated": datetime.now().isoformat()
            }
        }

    # 🔧 Helper methods for social signals calculation

    def _calculate_engagement_score(self, interactions):
        """Calculate normalized engagement score"""
        upvotes = interactions.get("upvotes", 0)
        comments = interactions.get("comments", 0)
        shares = interactions.get("shares", 0)
        likes = interactions.get("likes", 0)
        retweets = interactions.get("retweets", 0)
        views = interactions.get("views", 1)  # Avoid division by zero
        
        # Weighted engagement formula
        engagement_points = (
            upvotes * 1.0 +
            likes * 1.0 +
            comments * 2.0 +
            shares * 3.0 +
            retweets * 2.5
        )
        
        engagement_rate = engagement_points / max(views, 1)
        return min(engagement_rate * 100, 100)  # Cap at 100

    def _calculate_max_thread_length(self, item):
        """Calculate maximum thread depth"""
        replies = item.get("replies", [])
        if not replies:
            return 0
        
        def get_max_depth(replies_list, current_depth=0):
            if not replies_list:
                return current_depth
            max_depth = current_depth
            for reply in replies_list:
                nested_replies = reply.get("replies", [])
                depth = get_max_depth(nested_replies, current_depth + 1)
                max_depth = max(max_depth, depth)
            return max_depth
        
        return get_max_depth(replies)

    def _calculate_avg_response_depth(self, item):
        """Calculate average response depth"""
        replies = item.get("replies", [])
        if not replies:
            return 0.0
        
        def count_replies_at_depth(replies_list, depth=1):
            count = 0
            total_depth = 0
            for reply in replies_list:
                count += 1
                total_depth += depth
                nested_replies = reply.get("replies", [])
                nested_count, nested_total = count_replies_at_depth(nested_replies, depth + 1)
                count += nested_count
                total_depth += nested_total
            return count, total_depth
        
        total_replies, total_depth = count_replies_at_depth(replies)
        return total_depth / total_replies if total_replies > 0 else 0.0

    def _count_discussion_branches(self, item):
        """Count number of discussion branches"""
        replies = item.get("replies", [])
        
        def count_branches(replies_list):
            if not replies_list:
                return 0
            branches = len(replies_list)
            for reply in replies_list:
                nested_replies = reply.get("replies", [])
                branches += count_branches(nested_replies)
            return branches
        
        return count_branches(replies)

    def _calculate_validation_ratio(self, platform_data):
        """Calculate success/failure validation ratio"""
        successes = platform_data.get("success_confirmations", 0)
        failures = platform_data.get("failure_reports", 0)
        total = successes + failures
        return successes / total if total > 0 else 0.0
    
    def _calculate_content_length(self, item):
        """
        Calculate the length of content in an item.
        
        Args:
            item: Normalized item
            
        Returns:
            dict: Content length metrics
        """
        total_chars = 0
        total_words = 0
        
        # Process title
        title = item.get("content", {}).get("title", "")
        if title:
            total_chars += len(title)
            total_words += len(title.split())
        
        # Process body
        body = item.get("content", {}).get("body", "")
        if body:
            total_chars += len(body)
            total_words += len(body.split())
        
        # Process replies/comments
        if "replies" in item:
            for reply in item["replies"]:
                reply_content = reply.get("content", "")
                if reply_content:
                    total_chars += len(reply_content)
                    total_words += len(reply_content.split())
        
        return {
            "characters": total_chars,
            "words": total_words
        }
    
    def _estimate_reading_time(self, item):
        """
        Estimate reading time for content.
        
        Args:
            item: Normalized item
            
        Returns:
            int: Estimated reading time in seconds
        """
        # Average reading speed: 200-250 words per minute
        # Using 225 words per minute (3.75 words per second)
        words = item.get("derived", {}).get("content_length", {}).get("words", 0)
        if not words:
            words = self._calculate_content_length(item).get("words", 0)
        
        reading_time_seconds = int(words / 3.75)
        return max(1, reading_time_seconds)  # Minimum 1 second
    
    def _detect_language(self, item):
        """
        Detect the language of content.
        
        Args:
            item: Normalized item
            
        Returns:
            str: Detected language code
        """
        # This would typically use a language detection library
        # For simplicity, we'll just return English
        return "en"
    
    def _analyze_sentiment(self, item):
        """
        Analyze sentiment of content.
        
        Args:
            item: Normalized item
            
        Returns:
            dict: Sentiment analysis results
        """
        # This would typically use a sentiment analysis library
        # For simplicity, we'll return neutral sentiment
        return {
            "polarity": 0.0,  # -1.0 to 1.0
            "subjectivity": 0.5  # 0.0 to 1.0
        }
    
    def _extract_cross_references(self, item):
        """
        Extract cross-references to other platforms or content.
        
        Args:
            item: Normalized item
            
        Returns:
            dict: Cross-reference information
        """
        # This would typically involve URL extraction and classification
        # For simplicity, we'll return an empty dict
        return {
            "urls": [],
            "platforms": {}
        }
    
    def _extract_topics(self, item):
        """
        Extract topics from content.
        
        Args:
            item: Normalized item
            
        Returns:
            list: Extracted topics
        """
        # This would typically use topic modeling
        # For simplicity, we'll return an empty list
        return []
    
    def _extract_entities(self, item):
        """
        Extract entities from content.
        
        Args:
            item: Normalized item
            
        Returns:
            dict: Extracted entities by type
        """
        # This would typically use named entity recognition
        # For simplicity, we'll return empty categories
        return {
            "persons": [],
            "organizations": [],
            "locations": [],
            "products": [],
            "other": []
        }
    
    def _analytical_stage(self, item, platform):
        """
        Process item through analytical stage.
        
        Args:
            item: Enriched item
            platform: Source platform
            
        Returns:
            dict: Analytical item
        """
        # Create a copy of the enriched item
        analytical_item = {**item}
        
        # Generate aggregations
        analytical_item["aggregations"] = self._generate_aggregations(item)
        
        # Generate time-series projections
        analytical_item["time_series"] = self._generate_time_series(item)
        
        # Add analytical metadata
        analytical_item["_analytical_metadata"] = {
            "analytical_time": datetime.now().isoformat(),
            "analytical_schema_version": data_collection_config.SCHEMA_VERSIONS["analytical"]
        }
        
        # Record stage completion
        self._record_transformation_stage(item.get("id"), "analytical")
        
        return analytical_item
    
    def _generate_aggregations(self, item):
        """
        Generate aggregations from item data.
        
        Args:
            item: Enriched item
            
        Returns:
            dict: Aggregation results
        """
        # This would typically involve statistical analysis
        # For simplicity, we'll return basic aggregations
        return {
            "interaction_counts": item.get("interactions", {}),
            "content_summary": {
                "length": item.get("derived", {}).get("content_length", {}),
                "reading_time": item.get("derived", {}).get("reading_time", 0)
            }
        }
    
    def _generate_time_series(self, item):
        """
        Generate time-series projections.
        
        Args:
            item: Enriched item
            
        Returns:
            dict: Time-series data
        """
        # This would typically involve time-based analysis
        # For simplicity, we'll return empty time series
        return {
            "interaction_timeline": [],
            "engagement_forecast": []
        }
    
    def _record_transformation_start(self, item_id, platform):
        """
        Record transformation pipeline start.
        
        Args:
            item_id: Item identifier
            platform: Source platform
        """
        if not item_id:
            return
            
        self.transformation_history[item_id] = {
            "platform": platform,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "stages": {},
            "error": None
        }
    
    def _record_transformation_stage(self, item_id, stage):
        """
        Record completion of a transformation stage.
        
        Args:
            item_id: Item identifier
            stage: Stage name
        """
        if not item_id or item_id not in self.transformation_history:
            return
            
        self.transformation_history[item_id]["stages"][stage] = {
            "completed_at": datetime.now().isoformat()
        }
    
    def _record_transformation_completion(self, item_id, platform):
        """
        Record transformation pipeline completion.
        
        Args:
            item_id: Item identifier
            platform: Source platform
        """
        if not item_id or item_id not in self.transformation_history:
            return
            
        self.transformation_history[item_id]["end_time"] = datetime.now().isoformat()
    
    def _record_transformation_error(self, item_id, platform, error_message):
        """
        Record transformation pipeline error.
        
        Args:
            item_id: Item identifier
            platform: Source platform
            error_message: Error message
        """
        if not item_id or item_id not in self.transformation_history:
            return
            
        self.transformation_history[item_id]["error"] = error_message
        self.transformation_history[item_id]["end_time"] = datetime.now().isoformat()

    # 🆕 BENCHMARK PHASE: Update methods for post-testing data population

    def update_testing_history(self, item_id, test_results):
        """
        🆕 Update testing history after benchmark testing
        
        Args:
            item_id: Vulnerability item ID
            test_results: Results from benchmark testing
            
        Returns:
            bool: Success status
        """
        # This would be called from your benchmark orchestrator
        item = self.storage_manager.get_vulnerability(item_id)
        if not item:
            logger.error(f"Item {item_id} not found for testing history update")
            return False
        
        testing_history = item.get("testing_history", {})
        
        # Ensure structure exists
        if "tests_conducted" not in testing_history:
            testing_history = self._initialize_testing_history(item)
        
        # Add new test results
        test_record = {
            "test_id": test_results.get("test_id", f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            "timestamp": datetime.now().isoformat(),
            "models_tested": test_results.get("models", []),
            "transformations": test_results.get("transformations", []),
            "success_rate": test_results.get("overall_success_rate", 0.0),
            "individual_results": test_results.get("individual_results", []),
            "statistical_summary": test_results.get("statistical_summary", {}),
            "details": test_results
        }
        
        testing_history["tests_conducted"].append(test_record)
        
        # Update resilience over time tracking
        resilience_entry = {
            "timestamp": datetime.now().isoformat(),
            "success_rate": test_results.get("overall_success_rate", 0.0),
            "models_affected": len(test_results.get("models", [])),
            "transformation_resistance": test_results.get("transformation_resistance", {}),
            "mitigation_bypassed": test_results.get("mitigation_bypassed", False)
        }
        testing_history["resilience_over_time"].append(resilience_entry)
        
        # Track adaptation attempts (if the vulnerability was modified/evolved)
        if test_results.get("is_adaptation", False):
            adaptation_entry = {
                "timestamp": datetime.now().isoformat(),
                "adaptation_type": test_results.get("adaptation_type", "unknown"),
                "original_success_rate": test_results.get("original_success_rate", 0.0),
                "adapted_success_rate": test_results.get("overall_success_rate", 0.0),
                "improvement_delta": test_results.get("overall_success_rate", 0.0) - test_results.get("original_success_rate", 0.0)
            }
            testing_history["adaptation_attempts"].append(adaptation_entry)
        
        # Update mitigation effectiveness tracking
        for mitigation, effectiveness in test_results.get("mitigation_effectiveness", {}).items():
            if mitigation not in testing_history["mitigation_effectiveness"]:
                testing_history["mitigation_effectiveness"][mitigation] = []
            
            testing_history["mitigation_effectiveness"][mitigation].append({
                "timestamp": datetime.now().isoformat(),
                "effectiveness_score": effectiveness,
                "test_id": test_record["test_id"]
            })
        
        # Update temporal analysis
        testing_history["temporal_analysis"]["last_test_date"] = datetime.now().isoformat()
        if not testing_history["temporal_analysis"]["first_test_date"]:
            testing_history["temporal_analysis"]["first_test_date"] = datetime.now().isoformat()
        
        testing_history["temporal_analysis"]["test_frequency"] += 1
        testing_history["temporal_analysis"]["success_rate_over_time"].append({
            "timestamp": datetime.now().isoformat(),
            "success_rate": test_results.get("overall_success_rate", 0.0),
            "test_id": test_record["test_id"]
        })
        
        # Update metadata
        testing_history["metadata"]["last_updated"] = datetime.now().isoformat()
        testing_history["metadata"]["total_tests"] = len(testing_history["tests_conducted"])
        
        # Store updated testing history
        success = self.storage_manager.update_document_field(item_id, "testing_history", testing_history)
        
        if success:
            logger.info(f"Updated testing history for item {item_id} with test {test_record['test_id']}")
        else:
            logger.error(f"Failed to update testing history for item {item_id}")
            
        return success

    def update_execution_results(self, item_id, benchmark_results):
        """
        🆕 Update execution results after benchmark testing
        
        Args:
            item_id: Vulnerability item ID
            benchmark_results: Results from benchmark testing across models
            
        Returns:
            bool: Success status
        """
        item = self.storage_manager.get_vulnerability(item_id)
        if not item:
            logger.error(f"Item {item_id} not found for execution results update")
            return False
        
        execution_results = item.get("execution_results", {})
        
        # Ensure structure exists
        if "model_results" not in execution_results:
            execution_results = self._initialize_execution_results(item)
        
        # Update model results
        for model_name, results in benchmark_results.get("model_results", {}).items():
            execution_results["model_results"][model_name] = {
                "timestamp": datetime.now().isoformat(),
                "success_rate": results.get("success_rate", 0.0),
                "total_attempts": results.get("total_attempts", 0),
                "successful_attempts": results.get("successful_attempts", 0),
                "failed_attempts": results.get("failed_attempts", 0),
                "partial_successes": results.get("partial_successes", 0),
                "blocked_attempts": results.get("blocked_attempts", 0),
                "transformation_results": results.get("transformation_results", {}),
                "response_classifications": results.get("response_classifications", {}),
                "error_rate": results.get("error_rate", 0.0),
                "average_response_time": results.get("average_response_time", 0.0),
                "detailed_results": results.get("detailed_results", [])
            }
        
        # Update platform analysis (cross-model comparison by transformation type)
        execution_results["platform_analysis"] = self._analyze_cross_platform_performance(benchmark_results)
        
        # Update cross-model comparison
        execution_results["cross_model_comparison"] = self._generate_cross_model_comparison(benchmark_results)
        
        # Update effectiveness metrics
        execution_results["effectiveness_metrics"].update({
            "overall_success_rate": benchmark_results.get("overall_success_rate", 0.0),
            "model_specific_rates": {
                model: results.get("success_rate", 0.0) 
                for model, results in benchmark_results.get("model_results", {}).items()
            },
            "transformation_effectiveness": benchmark_results.get("transformation_effectiveness", {}),
            "consistency_score": self._calculate_consistency_score(benchmark_results),
            "robustness_score": self._calculate_robustness_score(benchmark_results)
        })
        
        # Update statistical analysis
        execution_results["statistical_analysis"] = self._perform_statistical_analysis(benchmark_results)
        
        # Update metadata
        execution_results["metadata"].update({
            "total_tests": benchmark_results.get("total_tests", 0),
            "models_tested": list(benchmark_results.get("model_results", {}).keys()),
            "transformations_tested": benchmark_results.get("transformations_tested", []),
            "last_updated": datetime.now().isoformat(),
            "benchmark_version": benchmark_results.get("benchmark_version", "1.0"),
            "test_duration": benchmark_results.get("test_duration", 0)
        })
        
        # Store updated execution results
        success = self.storage_manager.update_document_field(item_id, "execution_results", execution_results)
        
        if success:
            logger.info(f"Updated execution results for item {item_id}")
        else:
            logger.error(f"Failed to update execution results for item {item_id}")
            
        return success

    def _analyze_cross_platform_performance(self, benchmark_results):
        """
        Analyze performance differences across platforms/models
        """
        model_results = benchmark_results.get("model_results", {})
        
        if len(model_results) < 2:
            return {"insufficient_data": True}
        
        # Calculate variance in success rates
        success_rates = [results.get("success_rate", 0.0) for results in model_results.values()]
        variance = sum((rate - sum(success_rates) / len(success_rates)) ** 2 for rate in success_rates) / len(success_rates)
        
        # Find best and worst performing models
        best_model = max(model_results.items(), key=lambda x: x[1].get("success_rate", 0.0))
        worst_model = min(model_results.items(), key=lambda x: x[1].get("success_rate", 0.0))
        
        return {
            "performance_variance": variance,
            "best_performing_model": {
                "name": best_model[0],
                "success_rate": best_model[1].get("success_rate", 0.0)
            },
            "worst_performing_model": {
                "name": worst_model[0],
                "success_rate": worst_model[1].get("success_rate", 0.0)
            },
            "performance_gap": best_model[1].get("success_rate", 0.0) - worst_model[1].get("success_rate", 0.0),
            "model_count": len(model_results)
        }

    def _generate_cross_model_comparison(self, benchmark_results):
        """
        Generate detailed cross-model comparison metrics
        """
        model_results = benchmark_results.get("model_results", {})
        
        comparison = {
            "pairwise_comparisons": {},
            "ranking": [],
            "clustering": {},
            "transformation_sensitivity": {}
        }
        
        # Generate pairwise comparisons
        models = list(model_results.keys())
        for i, model_a in enumerate(models):
            for model_b in models[i+1:]:
                rate_a = model_results[model_a].get("success_rate", 0.0)
                rate_b = model_results[model_b].get("success_rate", 0.0)
                
                comparison["pairwise_comparisons"][f"{model_a}_vs_{model_b}"] = {
                    "difference": abs(rate_a - rate_b),
                    "relative_difference": abs(rate_a - rate_b) / max(rate_a, rate_b, 0.01),
                    "winner": model_a if rate_a > rate_b else model_b,
                    "significance": "high" if abs(rate_a - rate_b) > 0.2 else "medium" if abs(rate_a - rate_b) > 0.1 else "low"
                }
        
        # Generate ranking
        comparison["ranking"] = sorted(
            models, 
            key=lambda m: model_results[m].get("success_rate", 0.0), 
            reverse=True
        )
        
        # Analyze transformation sensitivity per model
        for model, results in model_results.items():
            transformation_results = results.get("transformation_results", {})
            if transformation_results:
                baseline_rate = transformation_results.get("baseline", {}).get("success_rate", 0.0)
                
                sensitivities = {}
                for transform, transform_results in transformation_results.items():
                    if transform != "baseline":
                        transform_rate = transform_results.get("success_rate", 0.0)
                        sensitivity = abs(baseline_rate - transform_rate)
                        sensitivities[transform] = sensitivity
                
                comparison["transformation_sensitivity"][model] = {
                    "average_sensitivity": sum(sensitivities.values()) / len(sensitivities) if sensitivities else 0.0,
                    "max_sensitivity": max(sensitivities.values()) if sensitivities else 0.0,
                    "most_sensitive_to": max(sensitivities.items(), key=lambda x: x[1])[0] if sensitivities else None,
                    "individual_sensitivities": sensitivities
                }
        
        return comparison

    def _calculate_consistency_score(self, benchmark_results):
        """
        Calculate how consistent the vulnerability is across models
        """
        model_results = benchmark_results.get("model_results", {})
        
        if len(model_results) < 2:
            return 0.0
        
        success_rates = [results.get("success_rate", 0.0) for results in model_results.values()]
        mean_rate = sum(success_rates) / len(success_rates)
        
        # Calculate coefficient of variation (inverted for consistency score)
        if mean_rate == 0:
            return 0.0
        
        variance = sum((rate - mean_rate) ** 2 for rate in success_rates) / len(success_rates)
        std_dev = variance ** 0.5
        cv = std_dev / mean_rate
        
        # Convert to consistency score (0-100, where 100 is perfectly consistent)
        consistency_score = max(0, 100 - (cv * 100))
        return consistency_score

    def _calculate_robustness_score(self, benchmark_results):
        """
        Calculate how robust the vulnerability is to different transformations
        """
        transformation_effectiveness = benchmark_results.get("transformation_effectiveness", {})
        
        if not transformation_effectiveness:
            return 0.0
        
        # Average effectiveness across all transformations
        effectiveness_scores = list(transformation_effectiveness.values())
        avg_effectiveness = sum(effectiveness_scores) / len(effectiveness_scores)
        
        # Robustness is how well the vulnerability maintains effectiveness across transformations
        return avg_effectiveness

    def _perform_statistical_analysis(self, benchmark_results):
        """
        Perform statistical analysis on benchmark results
        """
        model_results = benchmark_results.get("model_results", {})
        
        if len(model_results) < 2:
            return {"insufficient_data": True}
        
        success_rates = [results.get("success_rate", 0.0) for results in model_results.values()]
        
        # Basic descriptive statistics
        n = len(success_rates)
        mean = sum(success_rates) / n
        variance = sum((rate - mean) ** 2 for rate in success_rates) / n
        std_dev = variance ** 0.5
        
        # Confidence intervals (assuming normal distribution)
        # 95% confidence interval
        margin_of_error = 1.96 * (std_dev / (n ** 0.5)) if n > 1 else 0
        
        analysis = {
            "descriptive_stats": {
                "mean": mean,
                "median": sorted(success_rates)[n // 2],
                "std_dev": std_dev,
                "variance": variance,
                "min": min(success_rates),
                "max": max(success_rates),
                "range": max(success_rates) - min(success_rates)
            },
            "confidence_intervals": {
                "95_percent": {
                    "lower": max(0, mean - margin_of_error),
                    "upper": min(100, mean + margin_of_error),
                    "margin_of_error": margin_of_error
                }
            },
            "effect_sizes": {
                "cohens_d": self._calculate_cohens_d(success_rates),
                "effect_magnitude": self._interpret_effect_size(self._calculate_cohens_d(success_rates))
            },
            "distribution_analysis": {
                "is_normal": self._test_normality(success_rates),
                "outliers": self._detect_outliers(success_rates)
            }
        }
        
        return analysis

    def _calculate_cohens_d(self, rates):
        """Calculate Cohen's d effect size"""
        if len(rates) < 2:
            return 0.0
        
        # For simplicity, compare against a baseline of 50%
        baseline = 50.0
        mean = sum(rates) / len(rates)
        variance = sum((rate - mean) ** 2 for rate in rates) / len(rates)
        std_dev = variance ** 0.5
        
        if std_dev == 0:
            return 0.0
        
        return (mean - baseline) / std_dev

    def _interpret_effect_size(self, cohens_d):
        """Interpret Cohen's d effect size"""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def _test_normality(self, rates):
        """Simple normality test (Shapiro-Wilk would be better but this is basic)"""
        if len(rates) < 3:
            return None
        
        # Simple skewness test
        n = len(rates)
        mean = sum(rates) / n
        variance = sum((rate - mean) ** 2 for rate in rates) / n
        std_dev = variance ** 0.5
        
        if std_dev == 0:
            return True
        
        # Calculate skewness
        skewness = sum((rate - mean) ** 3 for rate in rates) / (n * std_dev ** 3)
        
        # Roughly normal if skewness is between -1 and 1
        return abs(skewness) < 1.0

    def _detect_outliers(self, rates):
        """Detect outliers using IQR method"""
        if len(rates) < 4:
            return []
        
        sorted_rates = sorted(rates)
        n = len(sorted_rates)
        q1_idx = n // 4
        q3_idx = 3 * n // 4
        
        q1 = sorted_rates[q1_idx]
        q3 = sorted_rates[q3_idx]
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = [rate for rate in rates if rate < lower_bound or rate > upper_bound]
        return outliers

    def get_transformation_history(self, item_id):
        """
        Get the transformation history for an item
        
        Args:
            item_id: Item identifier
            
        Returns:
            dict: Transformation history or None if not found
        """
        return self.transformation_history.get(item_id)

    def clear_transformation_history(self, item_id=None):
        """
        Clear transformation history for a specific item or all items
        
        Args:
            item_id: Item identifier (if None, clears all history)
        """
        if item_id:
            if item_id in self.transformation_history:
                del self.transformation_history[item_id]
        else:
            self.transformation_history.clear()

    def get_pipeline_statistics(self):
        """
        Get statistics about the transformation pipeline performance
        
        Returns:
            dict: Pipeline statistics
        """
        if not self.transformation_history:
            return {"message": "No transformation history available"}
        
        completed_transformations = [
            history for history in self.transformation_history.values()
            if history.get("end_time") and not history.get("error")
        ]
        
        failed_transformations = [
            history for history in self.transformation_history.values()
            if history.get("error")
        ]
        
        # Calculate average processing time for completed transformations
        processing_times = []
        for history in completed_transformations:
            try:
                start = datetime.fromisoformat(history["start_time"])
                end = datetime.fromisoformat(history["end_time"])
                duration = (end - start).total_seconds()
                processing_times.append(duration)
            except Exception:
                continue
        
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        # Count items by platform
        platform_counts = {}
        for history in self.transformation_history.values():
            platform = history.get("platform", "unknown")
            platform_counts[platform] = platform_counts.get(platform, 0) + 1
        
        return {
            "total_items_processed": len(self.transformation_history),
            "successful_transformations": len(completed_transformations),
            "failed_transformations": len(failed_transformations),
            "success_rate": len(completed_transformations) / len(self.transformation_history) * 100 if self.transformation_history else 0,
            "average_processing_time_seconds": avg_processing_time,
            "platform_distribution": platform_counts,
            "error_types": [history.get("error") for history in failed_transformations if history.get("error")]
        }
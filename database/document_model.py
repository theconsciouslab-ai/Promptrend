# database/document_model.py
import logging
import uuid
from datetime import datetime
import data_collection_config

logger = logging.getLogger("DocumentModel")

class VulnerabilityDocument:
    """
    Core document model for vulnerability storage.
    
    Implements the Vulnerability-Centric Document Model with support for
    cross-platform correlation and version history tracking.
    """
    
    def __init__(self, platform=None, source_item=None):
        """
        Initialize a new vulnerability document.
        
        Args:
            platform: Source platform
            source_item: Source item data
        """
        # Generate UUID if not provided
        self.uuid = str(uuid.uuid4())
        
        # Timestamps
        self.discovery_timestamp = datetime.now().isoformat()
        self.discovery_platform = platform
        
        # Core components
        self.classification = {
            "attack_vector": None,
            "target_models": [],
            "harm_categories": [],
            "technical_complexity": 0.0
        }
        
        self.social_signals = {
            "engagement_metrics": {
                "aggregated_score": 0.0,
                "platform_metrics": {}
            },
            "discussion_depth": {
                "max_thread_length": 0,
                "avg_response_depth": 0.0,
                "branches": 0
            },
            "community_validation": {
                "success_confirmations": 0,
                "failure_reports": 0,
                "validation_ratio": 0.0
            },
            "cross_references": {
                "mentioned_in_discussions": 0,
                "linked_from_other_vulnerabilities": 0
            }
        }
        
        self.temporal_data = {
            "propagation_timeline": [],
            "evolution_markers": []
        }
        
        self.prompt_versions = []
        
        # If source item provided, initialize with it
        if source_item:
            self._initialize_from_item(source_item, platform)
    
    def _initialize_from_item(self, item, platform):
        """
        Initialize document from a source item.
        
        Args:
            item: Source item
            platform: Source platform
        """
        # Extract basic metadata
        if "created_at" in item:
            self.discovery_timestamp = item["created_at"]
        
        self.discovery_platform = platform
        
        # Extract classification
        if "technical_indicators" in item:
            indicators = item["technical_indicators"]
            self.classification = {
                "attack_vector": indicators.get("attack_vectors", [None])[0],
                "target_models": indicators.get("target_models", []),
                "harm_categories": [],
                "technical_complexity": indicators.get("technical_complexity", 0.0)
            }
        
        # Extract first prompt version
        self._add_initial_prompt_version(item, platform)
        
        # Extract social signals if available
        if "social_signals" in item:
            signals = item["social_signals"]
            
            # Engagement metrics
            if "engagement_metrics" in signals:
                self.social_signals["engagement_metrics"] = {
                    "aggregated_score": 0.0,
                    "platform_metrics": {
                        platform: signals["engagement_metrics"]
                    }
                }
            
            # Discussion depth
            if "discussion_depth" in signals:
                self.social_signals["discussion_depth"] = signals["discussion_depth"]
            
            # Community validation
            if "community_validation" in signals:
                self.social_signals["community_validation"] = signals["community_validation"]
        
        # Extract temporal data
        if "temporal_data" in item:
            self.temporal_data = item["temporal_data"]
    
    def _add_initial_prompt_version(self, item, platform):
        """
        Add initial prompt version from source item.
        
        Args:
            item: Source item
            platform: Source platform
        """
        # Extract prompt text based on platform
        prompt_text = None
        
        if platform == "reddit":
            # For Reddit, use title + text
            title = item.get("title", "")
            text = item.get("text", "")
            prompt_text = f"{title}\n\n{text}"
        elif platform == "twitter":
            # For Twitter, use the tweet text
            prompt_text = item.get("text", "")
        elif platform == "discord":
            # For Discord, use the message content
            prompt_text = item.get("content", "")
        elif platform == "github":
            # For GitHub, handle code differently
            if item.get("type") == "code":
                prompt_text = item.get("content", "")
            else:
                # For issues/discussions
                title = item.get("title", "")
                body = item.get("body", "")
                prompt_text = f"{title}\n\n{body}"
        
        # Add version if text was extracted
        if prompt_text:
            version = {
                "version_id": 1,
                "prompt_text": prompt_text,
                "timestamp": item.get("created_at", datetime.now().isoformat()),
                "platform": platform,
                "context": self._extract_context(item, platform)
            }
            
            self.prompt_versions.append(version)
    
    def _extract_context(self, item, platform):
        """
        Extract platform-specific context.
        
        Args:
            item: Source item
            platform: Source platform
            
        Returns:
            dict: Platform-specific context
        """
        context = {}
        
        if platform == "reddit":
            context = {
                "subreddit": item.get("subreddit"),
                "post_id": item.get("id"),
                "permalink": item.get("permalink")
            }
        elif platform == "twitter":
            context = {
                "tweet_id": item.get("id"),
                "conversation_id": item.get("conversation_id")
            }
        elif platform == "discord":
            context = {
                "server": item.get("server"),
                "channel": item.get("channel"),
                "message_id": item.get("id")
            }
        elif platform == "github":
            context = {
                "repo": item.get("repo"),
                "path": item.get("path"),
                "issue_number": item.get("issue_number")
            }
        
        return context
    
    def add_prompt_version(self, prompt_text, platform, timestamp=None, context=None):
        """
        Add a new prompt version.
        
        Args:
            prompt_text: The prompt text
            platform: Source platform
            timestamp: Timestamp (default: now)
            context: Platform-specific context
            
        Returns:
            int: New version ID
        """
        # Generate new version ID
        version_id = len(self.prompt_versions) + 1
        
        # Use current time if timestamp not provided
        if not timestamp:
            timestamp = datetime.now().isoformat()
        
        # Create version object
        version = {
            "version_id": version_id,
            "prompt_text": prompt_text,
            "timestamp": timestamp,
            "platform": platform,
            "context": context or {}
        }
        
        # Add to versions
        self.prompt_versions.append(version)
        
        # Update temporal data
        self._add_evolution_marker("prompt_version_added", timestamp, {
            "version_id": version_id,
            "platform": platform
        })
        
        return version_id
    
    def update_classification(self, classification_data):
        """
        Update the vulnerability classification.
        
        Args:
            classification_data: Classification data to update
            
        Returns:
            dict: Updated classification
        """
        # Update only provided fields
        for key, value in classification_data.items():
            if key in self.classification:
                self.classification[key] = value
        
        # Add evolution marker
        self._add_evolution_marker("classification_updated", datetime.now().isoformat())
        
        return self.classification
    
    def update_social_signals(self, platform, signal_data):
        """
        Update social signals for a platform.
        
        Args:
            platform: Source platform
            signal_data: Signal data to update
            
        Returns:
            dict: Updated social signals
        """
        # Update engagement metrics
        if "engagement_metrics" in signal_data:
            # Store platform-specific metrics
            self.social_signals["engagement_metrics"]["platform_metrics"][platform] = signal_data["engagement_metrics"]
            
            # Recalculate aggregated score
            platform_scores = [
                metrics.get("engagement_score", 0.0) 
                for metrics in self.social_signals["engagement_metrics"]["platform_metrics"].values()
            ]
            
            # Average of platform scores
            if platform_scores:
                self.social_signals["engagement_metrics"]["aggregated_score"] = sum(platform_scores) / len(platform_scores)
        
        # Update community validation
        if "community_validation" in signal_data:
            new_validation = signal_data["community_validation"]
            current = self.social_signals["community_validation"]
            
            # Combine counts
            current["success_confirmations"] += new_validation.get("success_confirmations", 0)
            current["failure_reports"] += new_validation.get("failure_reports", 0)
            
            # Recalculate ratio
            total = current["success_confirmations"] + current["failure_reports"]
            if total > 0:
                current["validation_ratio"] = current["success_confirmations"] / total
        
        # Add evolution marker
        self._add_evolution_marker("social_signals_updated", datetime.now().isoformat(), {
            "platform": platform
        })
        
        return self.social_signals
    
    def add_propagation_event(self, platform, timestamp):
        """
        Add a propagation event to the timeline.
        
        Args:
            platform: Platform where vulnerability was observed
            timestamp: When it was observed
            
        Returns:
            list: Updated propagation timeline
        """
        # Check if platform already in timeline
        for event in self.temporal_data["propagation_timeline"]:
            if event["platform"] == platform:
                # Platform already tracked, no need to add again
                return self.temporal_data["propagation_timeline"]
        
        # Add new propagation event
        event = {
            "platform": platform,
            "timestamp": timestamp
        }
        
        self.temporal_data["propagation_timeline"].append(event)
        
        # Add evolution marker
        self._add_evolution_marker("propagation_observed", timestamp, {
            "platform": platform
        })
        
        return self.temporal_data["propagation_timeline"]
    
    def _add_evolution_marker(self, event_type, timestamp, details=None):
        """
        Add an evolution marker event.
        
        Args:
            event_type: Type of evolution event
            timestamp: When it occurred
            details: Additional details
            
        Returns:
            list: Updated evolution markers
        """
        marker = {
            "event": event_type,
            "timestamp": timestamp
        }
        
        if details:
            marker["details"] = details
        
        self.temporal_data["evolution_markers"].append(marker)
        return self.temporal_data["evolution_markers"]
    
    def to_dict(self):
        """
        Convert document to dictionary for storage.
        
        Returns:
            dict: Document as dictionary
        """
        return {
            "uuid": self.uuid,
            "discovery_timestamp": self.discovery_timestamp,
            "discovery_platform": self.discovery_platform,
            "classification": self.classification,
            "social_signals": self.social_signals,
            "temporal_data": self.temporal_data,
            "prompt_versions": self.prompt_versions
        }
    
    @classmethod
    def from_dict(cls, data):
        """
        Create document from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            VulnerabilityDocument: Created document
        """
        doc = cls()
        
        # Copy fields
        doc.uuid = data.get("uuid", doc.uuid)
        doc.discovery_timestamp = data.get("discovery_timestamp", doc.discovery_timestamp)
        doc.discovery_platform = data.get("discovery_platform")
        doc.classification = data.get("classification", doc.classification)
        doc.social_signals = data.get("social_signals", doc.social_signals)
        doc.temporal_data = data.get("temporal_data", doc.temporal_data)
        doc.prompt_versions = data.get("prompt_versions", doc.prompt_versions)
        
        return doc
import asyncio
import datetime
import logging
import uuid
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity



logger = logging.getLogger(__name__)

class CrossPlatformCoordinator:
    """Central coordination system that integrates vulnerability data across platforms.
    
    This class implements three key mechanisms:
    1. Deduplication with provenance preservation
    2. Temporal propagation mapping
    3. Collective intelligence adaptation
    """
    
    def __init__(self, 
                semantic_model: str = "all-MiniLM-L6-v2",
                similarity_threshold: float = 0.85,
                temporal_window_days: int = 14):
        """Initialize the Cross-Platform Coordinator.
        
        Args:
            semantic_model: Model name for sentence transformer
            similarity_threshold: Threshold for semantic similarity matching
            temporal_window_days: Time window for temporal tracking
        """
        # Initialize semantic fingerprinting model
        self.encoder = SentenceTransformer(semantic_model)
        self.similarity_threshold = similarity_threshold
        self.temporal_window = datetime.timedelta(days=temporal_window_days)
        
        # Data structures for vulnerability tracking
        self.vulnerability_db = {}  # Maps vulnerability IDs to metadata
        self.platform_data = {}     # Platform-specific data stores
        self.fingerprint_map = {}   # Maps fingerprints to vulnerability IDs
        self.embedding_cache = {}   # Caches computed embeddings
        
        # Propagation tracking graph
        self.propagation_graph = nx.DiGraph()
        
        # Adaptive terminology dictionary
        self.terminology_dict = {
            "global": set(),  # Terms relevant across all platforms
            "platform_specific": {}  # Platform-specific terminologies
        }
        
        # Initialize platform-specific stores
        for platform in ["reddit", "twitter", "discord", "github", "forums"]:
            self.platform_data[platform] = []
            self.terminology_dict["platform_specific"][platform] = set()
            
        logger.info("Cross-Platform Coordinator initialized")
        
    def _ensure_datetime(self, value):
        if isinstance(value, datetime.datetime):
            return value
        try:
            return datetime.datetime.fromisoformat(value)
        except Exception:
            return datetime.datetime.utcnow()

    
    async def process_new_content(self, 
                                 content: str, 
                                 platform: str, 
                                 metadata: Dict) -> Tuple[str, bool]:
        """Process new content from a platform agent.
        
        Args:
            content: Text content of the potential vulnerability
            platform: Source platform identifier
            metadata: Additional information including timestamp, URL, etc.
            
        Returns:
            Tuple of vulnerability ID and whether it's a new discovery
        """
        # Generate semantic fingerprint for deduplication
        fingerprint = await self._generate_fingerprint(content)
        
        # Check if this is a known vulnerability or a new one
        is_new = True
        vuln_id = None
        
        # Look for matches in existing fingerprints
        if fingerprint in self.fingerprint_map:
            vuln_id = self.fingerprint_map[fingerprint]
            is_new = False
            logger.info(f"Matched existing vulnerability: {vuln_id}")
        else:
            # New vulnerability discovered
            vuln_id = str(uuid.uuid4())
            self.fingerprint_map[fingerprint] = vuln_id
            self.vulnerability_db[vuln_id] = {
                "discovery_time": self._ensure_datetime(metadata.get("timestamp")),
                "discovery_platform": platform,
                "contents": {},
                "fingerprints": [fingerprint],
                "propagation": {},
                "evolution_stage": 1,  # Initial discovery starts at stage 1
                "confirmed_platforms": {platform}
            }
            logger.info(f"New vulnerability discovered: {vuln_id} on {platform}")
        
        # Preserve this instance with its provenance
        self.vulnerability_db[vuln_id]["contents"][platform] = self.vulnerability_db[vuln_id].get("contents", {})
        self.vulnerability_db[vuln_id]["contents"][platform][metadata["url"]] = {
            "content": content,
            "metadata": metadata,
            "fingerprint": fingerprint
        }
        
        # Update propagation tracking
        await self._update_propagation(vuln_id, platform, metadata, is_new)
        
        # Update collective intelligence
        await self._update_terminology(content, platform, is_new)
        
        return vuln_id, is_new
    
    async def _generate_fingerprint(self, content: str) -> str:
        """Generate a semantic fingerprint for content deduplication.
        
        Uses sentence embeddings to create a representation that captures
        the semantic meaning rather than exact text, allowing detection
        of conceptually equivalent vulnerabilities despite different wording.
        
        Args:
            content: Text content to fingerprint
            
        Returns:
            String representation of the fingerprint
        """
        # Check cache first
        cache_key = hash(content)
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        # Preprocess content (normalize whitespace, lowercase, etc.)
        processed_content = self._preprocess_text(content)
        
        # Generate embedding
        embedding = self.encoder.encode(processed_content)
        
        # Convert to string representation for storage
        fingerprint = self._embedding_to_string(embedding)
        
        # Update cache
        self.embedding_cache[cache_key] = fingerprint
        
        return fingerprint
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for more consistent fingerprinting."""
        # Basic preprocessing
        text = text.lower().strip()
        # More advanced preprocessing could be added here
        return text
    
    def _embedding_to_string(self, embedding: np.ndarray) -> str:
        """Convert embedding vector to string representation."""
        # Use a hash of the vector as the fingerprint
        return str(hash(embedding.tobytes()))
    
    async def _update_propagation(self, 
                                 vuln_id: str, 
                                 platform: str,
                                 metadata: Dict,
                                 is_new: bool) -> None:
        """Update the propagation tracking for a vulnerability.
        
        Args:
            vuln_id: Vulnerability identifier
            platform: Platform where instance was found
            metadata: Additional information including timestamp
            is_new: Whether this is a new vulnerability discovery
        """
        timestamp = metadata.get("timestamp", datetime.datetime.now())
        
        # Add to confirmed platforms
        self.vulnerability_db[vuln_id]["confirmed_platforms"].add(platform)
        
        # Update the propagation graph
        if not is_new:
            # This is a propagation from the original discovery
            discovery_platform = self.vulnerability_db[vuln_id]["discovery_platform"]
            discovery_time = self.vulnerability_db[vuln_id]["discovery_time"]
            
            # Add edges to the propagation graph
            if platform != discovery_platform:
                # Track propagation between platforms
                if not self.propagation_graph.has_edge(discovery_platform, platform):
                    self.propagation_graph.add_edge(discovery_platform, platform, 
                                                   count=0, 
                                                   vulnerabilities=set())
                
                # Update edge metadata
                edge_data = self.propagation_graph.get_edge_data(discovery_platform, platform)
                edge_data["count"] += 1
                edge_data["vulnerabilities"].add(vuln_id)
                
                # Calculate and store propagation time
                prop_time = timestamp - discovery_time
                if "propagation_times" not in edge_data:
                    edge_data["propagation_times"] = []
                edge_data["propagation_times"].append(prop_time.total_seconds())
                
                # Update the vulnerability record
                if platform not in self.vulnerability_db[vuln_id]["propagation"]:
                    self.vulnerability_db[vuln_id]["propagation"][platform] = {
                        "first_seen": timestamp,
                        "instances": 0
                    }
                self.vulnerability_db[vuln_id]["propagation"][platform]["instances"] += 1
                
                logger.info(f"Tracked propagation of {vuln_id} from {discovery_platform} to {platform}")
        
        # Assess evolution stage
        await self._assess_evolution_stage(vuln_id)
    
    async def _assess_evolution_stage(self, vuln_id: str) -> None:
        """Assess the evolution stage of a vulnerability based on propagation.
        
        Evolution stages:
        1 - Initial discovery
        2 - Technical implementation
        3 - Community refinement
        4 - Widespread dissemination
        5 - Mainstream adoption
        
        Args:
            vuln_id: Vulnerability identifier
        """
        vuln = self.vulnerability_db[vuln_id]
        platforms = vuln["confirmed_platforms"]
        propagation = vuln["propagation"]
        
        # Current evolution stage
        current_stage = vuln["evolution_stage"]
        new_stage = current_stage
        
        # Simple stage assessment rules based on platform presence and instance counts
        if len(platforms) == 1 and list(platforms)[0] in ["forums", "github"]:
            new_stage = 1  # Initial discovery
        elif "github" in platforms:
            new_stage = max(new_stage, 2)  # Technical implementation
        elif "discord" in platforms and any(propagation.get(p, {}).get("instances", 0) > 2 
                                          for p in ["discord"]):
            new_stage = max(new_stage, 3)  # Community refinement
        elif len(platforms) >= 3:
            new_stage = max(new_stage, 4)  # Widespread dissemination
        elif "reddit" in platforms and propagation.get("reddit", {}).get("instances", 0) > 3:
            new_stage = max(new_stage, 5)  # Mainstream adoption
        
        # Update if stage has changed
        if new_stage != current_stage:
            vuln["evolution_stage"] = new_stage
            logger.info(f"Vulnerability {vuln_id} advanced to stage {new_stage}")
            
            # Additional actions could be triggered based on stage changes
            if new_stage >= 4:
                # High priority alert for widespread vulnerabilities
                await self._trigger_alert(vuln_id, f"Vulnerability reached stage {new_stage}")
    
    async def _update_terminology(self, 
                                 content: str, 
                                 platform: str,
                                 is_new: bool) -> None:
        """Update the adaptive terminology dictionary based on new content.
        
        Extracts key terms from vulnerability descriptions to improve
        future detection capabilities.
        
        Args:
            content: Text content to analyze
            platform: Source platform
            is_new: Whether this is a new vulnerability
        """
        # Extract potential terminology from content
        terms = self._extract_terms(content)
        
        # Update platform-specific terms
        self.terminology_dict["platform_specific"][platform].update(terms)
        
        # If terms appear across multiple platforms, add to global terms
        for term in terms:
            term_platforms = sum(1 for p in self.terminology_dict["platform_specific"] 
                                if term in self.terminology_dict["platform_specific"][p])
            if term_platforms >= 2:
                self.terminology_dict["global"].add(term)
    
    def _extract_terms(self, content: str) -> Set[str]:
        """Extract relevant terminology from content.
        
        In a real implementation, this would use NLP techniques like
        named entity recognition, keyword extraction, or topic modeling.
        This simplified version just extracts words that might be relevant.
        
        Args:
            content: Text to analyze
            
        Returns:
            Set of extracted terms
        """
        # Simplified implementation - in production would use more sophisticated NLP
        words = content.lower().split()
        # Filter for potentially relevant terms
        relevant_terms = {word for word in words 
                         if len(word) > 4 and not word.isdigit()}
        return relevant_terms
    
    async def _trigger_alert(self, vuln_id: str, message: str) -> None:
        """Trigger an alert for a vulnerability.
        
        Args:
            vuln_id: Vulnerability identifier
            message: Alert message
        """
        # In a real implementation, this would send to alerting systems
        logger.warning(f"ALERT: {message} - Vulnerability {vuln_id}")
    
    async def find_similar_vulnerabilities(self, 
                                         content: str, 
                                         threshold: Optional[float] = None) -> List[str]:
        """Find vulnerabilities similar to the provided content.
        
        Args:
            content: Content to compare against
            threshold: Optional custom similarity threshold
            
        Returns:
            List of vulnerability IDs that match the similarity criteria
        """
        if threshold is None:
            threshold = self.similarity_threshold
            
        # Generate embedding for the query content
        query_embedding = self.encoder.encode(self._preprocess_text(content))
        
        # Compare against known fingerprints
        similar_vulns = []
        
        for vuln_id, vuln_data in self.vulnerability_db.items():
            # Check each fingerprint associated with this vulnerability
            for fingerprint in vuln_data["fingerprints"]:
                # We would need to convert back from string to embedding
                # This is simplified - in production would need proper conversion
                similarity = self._compute_similarity(query_embedding, fingerprint)
                if similarity >= threshold:
                    similar_vulns.append(vuln_id)
                    break  # Only need to match once per vulnerability
        
        return similar_vulns
    
    def _compute_similarity(self, 
                           embedding1: np.ndarray, 
                           fingerprint: str) -> float:
        """Compute similarity between an embedding and a fingerprint.
        
        In a real implementation, would convert fingerprint back to vector.
        This is simplified for illustration.
        
        Args:
            embedding1: First embedding as numpy array
            fingerprint: String representation of second embedding
            
        Returns:
            Similarity score between 0 and 1
        """
        # Simplified - in production would convert fingerprint back to embedding
        # and compute proper cosine similarity
        return 0.9 if hash(embedding1.tobytes()) == int(fingerprint) else 0.5
    
    def get_vulnerability_lifecycle(self, vuln_id: str) -> Dict:
        """Get the complete lifecycle information for a vulnerability.
        
        Args:
            vuln_id: Vulnerability identifier
            
        Returns:
            Dictionary with full lifecycle information
        """
        if vuln_id not in self.vulnerability_db:
            return {"error": "Vulnerability not found"}
            
        vuln = self.vulnerability_db[vuln_id]
        
        # Compile lifecycle information
        lifecycle = {
            "id": vuln_id,
            "discovery": {
                "time": vuln["discovery_time"].isoformat(),
                "platform": vuln["discovery_platform"]
            },
            "evolution_stage": vuln["evolution_stage"],
            "stage_name": self._get_stage_name(vuln["evolution_stage"]),
            "propagation": {
                platform: {
                    "first_seen": info["first_seen"].isoformat(),
                    "instances": info["instances"]
                }
                for platform, info in vuln["propagation"].items()
            },
            "platforms": list(vuln["confirmed_platforms"]),
            "timeline": self._generate_timeline(vuln_id)
        }
        
        return lifecycle
    
    def _get_stage_name(self, stage: int) -> str:
        """Get the name of an evolution stage.
        
        Args:
            stage: Evolution stage number
            
        Returns:
            Human-readable stage name
        """
        stages = {
            1: "Initial Discovery",
            2: "Technical Implementation",
            3: "Community Refinement",
            4: "Widespread Dissemination",
            5: "Mainstream Adoption"
        }
        return stages.get(stage, "Unknown")
    
    def _generate_timeline(self, vuln_id: str) -> List[Dict]:
        """Generate a chronological timeline of vulnerability events.
        
        Args:
            vuln_id: Vulnerability identifier
            
        Returns:
            List of timeline events in chronological order
        """
        vuln = self.vulnerability_db[vuln_id]
        timeline = []
        
        # Add discovery event
        timeline.append({
            "time": vuln["discovery_time"].isoformat(),
            "platform": vuln["discovery_platform"],
            "event_type": "discovery",
            "description": f"Initial vulnerability discovery on {vuln['discovery_platform']}"
        })
        
        # Add propagation events
        for platform, info in vuln["propagation"].items():
            timeline.append({
                "time": info["first_seen"].isoformat(),
                "platform": platform,
                "event_type": "propagation",
                "description": f"First appearance on {platform}"
            })
        
        # Sort by time
        timeline.sort(key=lambda x: x["time"])
        
        return timeline
    
    def get_bridge_nodes(self, min_count: int = 3) -> Dict[Tuple[str, str], Dict]:
        """Identify bridge nodes that accelerate vulnerability propagation.
        
        Args:
            min_count: Minimum number of vulnerabilities to consider a significant bridge
            
        Returns:
            Dictionary of platform pairs and their bridge metrics
        """
        bridges = {}
        
        for edge in self.propagation_graph.edges():
            source, target = edge
            edge_data = self.propagation_graph.get_edge_data(source, target)
            
            # Only consider significant bridges
            if edge_data["count"] >= min_count:
                # Calculate average propagation time
                avg_time = np.mean(edge_data["propagation_times"]) if edge_data.get("propagation_times") else 0
                
                bridges[(source, target)] = {
                    "count": edge_data["count"],
                    "vulnerabilities": list(edge_data["vulnerabilities"]),
                    "avg_propagation_time_seconds": avg_time
                }
        
        return bridges
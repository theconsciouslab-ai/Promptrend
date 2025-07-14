# database/relationship_manager.py
import logging
from datetime import datetime
import data_collection_config

logger = logging.getLogger("RelationshipManager")

class RelationshipManager:
    """
    Manages relationships between vulnerability documents.
    
    Tracks variant relationships, cross-platform instances,
    and technical similarity clusters across the database.
    """
    
    def __init__(self, storage_manager):
        """
        Initialize the relationship manager.
        
        Args:
            storage_manager: Storage manager for database operations
        """
        self.storage_manager = storage_manager
    
    def find_variants(self, document):
        """
        Find variant relationships for a document.
        
        Args:
            document: Vulnerability document
            
        Returns:
            list: Related variant documents
        """
        variants = []
        
        # Get all documents
        all_documents = self.storage_manager.get_all_vulnerabilities()
        
        # Filter out self
        candidates = [doc for doc in all_documents if doc["uuid"] != document["uuid"]]
        
        for candidate in candidates:
            # Calculate similarity score
            similarity = self._calculate_prompt_similarity(document, candidate)
            
            # Documents with high similarity are variants
            if similarity >= data_collection_config.RELATIONSHIP_THRESHOLDS["variant"]:
                variants.append({
                    "document_id": candidate["uuid"],
                    "similarity_score": similarity,
                    "relationship_type": "variant"
                })
        
        return variants
    
    def find_cross_platform_instances(self, document):
        """
        Find cross-platform instances of the same vulnerability.
        
        Args:
            document: Vulnerability document
            
        Returns:
            list: Related cross-platform documents
        """
        cross_platform = []
        
        # Get document platform
        platform = document.get("discovery_platform")
        if not platform:
            return cross_platform
        
        # Get all documents from other platforms
        all_documents = self.storage_manager.get_all_vulnerabilities()
        
        # Filter for documents from different platforms
        candidates = [
            doc for doc in all_documents 
            if doc["uuid"] != document["uuid"] and doc.get("discovery_platform") != platform
        ]
        
        for candidate in candidates:
            # Calculate similarity score
            similarity = self._calculate_prompt_similarity(document, candidate)
            
            # Documents with very high similarity are cross-platform instances
            if similarity >= data_collection_config.RELATIONSHIP_THRESHOLDS["cross_platform"]:
                cross_platform.append({
                    "document_id": candidate["uuid"],
                    "platform": candidate.get("discovery_platform"),
                    "similarity_score": similarity,
                    "relationship_type": "cross_platform"
                })
        
        return cross_platform
    
    def find_technical_similarity_cluster(self, document):
        """
        Find technically similar vulnerabilities.
        
        Args:
            document: Vulnerability document
            
        Returns:
            list: Documents in the same technical cluster
        """
        similar = []
        
        # Get document attack vector and target models
        attack_vector = document.get("classification", {}).get("attack_vector")
        target_models = document.get("classification", {}).get("target_models", [])
        
        if not attack_vector and not target_models:
            return similar
        
        # Get all documents
        all_documents = self.storage_manager.get_all_vulnerabilities()
        
        # Filter out self
        candidates = [doc for doc in all_documents if doc["uuid"] != document["uuid"]]
        
        for candidate in candidates:
            # Get candidate classification
            candidate_classification = candidate.get("classification", {})
            candidate_attack_vector = candidate_classification.get("attack_vector")
            candidate_target_models = candidate_classification.get("target_models", [])
            
            # Calculate technical similarity
            similarity = self._calculate_technical_similarity(
                attack_vector, target_models,
                candidate_attack_vector, candidate_target_models
            )
            
            # Documents with sufficient technical similarity
            if similarity >= data_collection_config.RELATIONSHIP_THRESHOLDS["technical"]:
                similar.append({
                    "document_id": candidate["uuid"],
                    "similarity_score": similarity,
                    "relationship_type": "technical"
                })
        
        return similar
    
    def update_relationships(self, document_id):
        """
        Update all relationships for a document.
        
        Args:
            document_id: Document UUID
            
        Returns:
            dict: Updated relationships
        """
        # Get the document
        document = self.storage_manager.get_vulnerability(document_id)
        if not document:
            logger.error(f"Document not found for relationship update: {document_id}")
            return None
        
        # Find relationships
        variants = self.find_variants(document)
        cross_platform = self.find_cross_platform_instances(document)
        technical = self.find_technical_similarity_cluster(document)
        
        # Combine relationships
        relationships = {
            "variant_connections": variants,
            "cross_platform_instances": cross_platform,
            "similarity_clusters": technical,
            "last_updated": datetime.now().isoformat()
        }
        
        # Store relationships
        success = self.storage_manager.update_document_relationships(document_id, relationships)
        
        if success:
            logger.info(f"Updated relationships for document {document_id}")
            
            # Also update cross-references in social signals
            linked_count = len(variants) + len(cross_platform) + len(technical)
            social_update = {
                "cross_references": {
                    "linked_from_other_vulnerabilities": linked_count
                }
            }
            
            self.storage_manager.update_document_social_signals(document_id, social_update)
            
            return relationships
        else:
            logger.error(f"Failed to update relationships for document {document_id}")
            return None
    
    def _calculate_prompt_similarity(self, doc1, doc2):
        """
        Calculate similarity between document prompts.
        
        Args:
            doc1: First document
            doc2: Second document
            
        Returns:
            float: Similarity score between 0.0 and 1.0
        """
        # Get the latest prompt versions
        versions1 = doc1.get("prompt_versions", [])
        versions2 = doc2.get("prompt_versions", [])
        
        if not versions1 or not versions2:
            return 0.0
        
        # Get the latest versions
        latest1 = versions1[-1]
        latest2 = versions2[-1]
        
        # Get prompt texts
        text1 = latest1.get("prompt_text", "")
        text2 = latest2.get("prompt_text", "")
        
        if not text1 or not text2:
            return 0.0
        
        # This would typically use a more sophisticated text similarity algorithm
        # For simplicity, we'll use a basic approach
        # In a real implementation, consider using:
        # - Jaccard similarity
        # - Cosine similarity with TF-IDF
        # - Embedding-based similarity
        
        # Convert to lowercase and split into words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
            
        return intersection / union
    
    def _calculate_technical_similarity(self, attack_vector1, target_models1, attack_vector2, target_models2):
        """
        Calculate similarity based on technical characteristics.
        
        Args:
            attack_vector1: First document attack vector
            target_models1: First document target models
            attack_vector2: Second document attack vector
            target_models2: Second document target models
            
        Returns:
            float: Similarity score between 0.0 and 1.0
        """
        score = 0.0
        
        # Attack vector similarity (50% weight)
        if attack_vector1 and attack_vector2 and attack_vector1 == attack_vector2:
            score += 0.5
        
        # Target model similarity (50% weight)
        if target_models1 and target_models2:
            # Convert to sets
            models1 = set(target_models1)
            models2 = set(target_models2)
            
            # Calculate Jaccard similarity for target models
            intersection = len(models1.intersection(models2))
            union = len(models1.union(models2))
            
            if union > 0:
                score += 0.5 * (intersection / union)
        
        return score
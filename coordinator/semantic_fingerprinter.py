from typing import Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticFingerprinter:
    """Generates semantic fingerprints for vulnerability deduplication.
    
    This class handles the generation of semantic fingerprints that capture
    the meaning rather than exact text, allowing detection of conceptually
    equivalent vulnerabilities across different platforms and expressions.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the fingerprinter.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.fingerprint_cache = {}
        
    def generate_fingerprint(self, text: str) -> Tuple[str, np.ndarray]:
        """Generate a semantic fingerprint for text.
        
        Args:
            text: Input text to fingerprint
            
        Returns:
            Tuple of (fingerprint_id, embedding_vector)
        """
        # Check cache first
        cache_key = hash(text)
        if cache_key in self.fingerprint_cache:
            return self.fingerprint_cache[cache_key]
        
        # Preprocess text
        processed_text = self._preprocess(text)
        
        # Generate embedding
        embedding = self.model.encode(processed_text)
        
        # Generate fingerprint ID
        fingerprint_id = self._embedding_to_id(embedding)
        
        # Cache result
        result = (fingerprint_id, embedding)
        self.fingerprint_cache[cache_key] = result
        
        return result
    
    def compare_texts(self, text1: str, text2: str) -> float:
        """Compare two texts semantically.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score from 0 to 1
        """
        # Generate embeddings
        _, embedding1 = self.generate_fingerprint(text1)
        _, embedding2 = self.generate_fingerprint(text2)
        
        # Compute similarity
        return self._compute_similarity(embedding1, embedding2)
    
    def compare_with_fingerprint(self, text: str, fingerprint_id: str) -> Optional[float]:
        """Compare text with an existing fingerprint ID.
        
        Args:
            text: Text to compare
            fingerprint_id: Existing fingerprint ID
            
        Returns:
            Similarity score if fingerprint exists, None otherwise
        """
        # Generate embedding for text
        _, embedding = self.generate_fingerprint(text)
        
        # In a real implementation, would retrieve the embedding for the fingerprint ID
        # from a database. This is simplified.
        stored_embedding = self._get_embedding_for_fingerprint(fingerprint_id)
        if stored_embedding is None:
            return None
            
        # Compute similarity
        return self._compute_similarity(embedding, stored_embedding)
    
    def _preprocess(self, text: str) -> str:
        """Preprocess text for more consistent fingerprinting.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Basic preprocessing
        text = text.lower().strip()
        
        # Additional preprocessing could include:
        # - Removing stop words
        # - Stemming/lemmatization
        # - Removing code-specific syntax
        # - Normalizing URLs, paths, etc.
        
        return text
    
    def _embedding_to_id(self, embedding: np.ndarray) -> str:
        """Convert embedding to a unique fingerprint ID.
        
        Args:
            embedding: Embedding vector
            
        Returns:
            Fingerprint ID string
        """
        # Use a hash of the vector as a simple ID
        # In production, would use a more robust method
        return f"fp_{hash(embedding.tobytes())}"
    
    def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score from 0 to 1
        """
        # Reshape for cosine_similarity function which expects 2D arrays
        e1 = embedding1.reshape(1, -1)
        e2 = embedding2.reshape(1, -1)
        
        # Compute cosine similarity
        return float(cosine_similarity(e1, e2)[0][0])
    
    def _get_embedding_for_fingerprint(self, fingerprint_id: str) -> Optional[np.ndarray]:
        """Get the embedding vector for a fingerprint ID.
        
        In a real implementation, this would retrieve from a database.
        This is a simplified placeholder.
        
        Args:
            fingerprint_id: Fingerprint ID
            
        Returns:
            Embedding vector if found, None otherwise
        """
        # Simplified placeholder
        # In production, would retrieve from a database
        return None
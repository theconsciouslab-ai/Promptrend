# collectors/keyword_monitor.py
import logging
import re
import json
from datetime import datetime
import data_collection_config

logger = logging.getLogger("KeywordMonitor")

class KeywordMonitor:
    """
    Monitors content for relevant keywords and evolves lexicon over time.
    
    Implements a three-layer keyword architecture:
    1. Core terms: Established vulnerability terminology
    2. Contextual modifiers: Syntax patterns indicating exploits
    3. Emergent terminology: Dynamically updated based on discoveries
    """
    
    def __init__(self):
        """Initialize the keyword monitor with lexicons from config."""
        # Core terminology from established taxonomies
        self.core_terms = set(data_collection_config.LEXICON["core_terms"])
        
        # Contextual modifiers (patterns that suggest exploits)
        self.contextual_patterns = data_collection_config.LEXICON["contextual_patterns"]
        
        # Emergent terminology (dynamically updated)
        self.emergent_terms = set(data_collection_config.LEXICON["emergent_terms"])
        
        # Track term effectiveness for evolution
        self.term_effectiveness = {}
        
        # Track timestamp for last lexicon update
        self.last_update = datetime.now()
        
        logger.info(f"Initialized keyword monitor with {len(self.core_terms)} core terms, "
                   f"{len(self.contextual_patterns)} contextual patterns, and "
                   f"{len(self.emergent_terms)} emergent terms")
    
    def check_relevance(self, text, platform):
        """
        Check content relevance based on keyword presence.
        
        Args:
            text: Content text to check
            platform: Source platform
            
        Returns:
            float: Relevance score between 0.0 and 1.0
        """
        if not text:
            return 0.0
            
        # Convert to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Initialize scores for each lexicon layer
        core_score = 0.0
        pattern_score = 0.0
        emergent_score = 0.0
        
        # Check core terminology (highest confidence)
        matches = []
        for term in self.core_terms:
            term_lower = term.lower()
            if term_lower in text_lower:
                matches.append(term)
                # Update effectiveness tracking
                self._update_term_effectiveness(term, "core")
                
        # Calculate score based on number of matches
        if matches:
            # Higher weight for core terms
            core_score = min(0.6, len(matches) * 0.15)
        
        # Check contextual patterns
        for pattern, weight in self.contextual_patterns.items():
            if re.search(pattern, text_lower):
                pattern_score += weight
                # Update effectiveness tracking
                self._update_term_effectiveness(pattern, "pattern")
        
        # Cap pattern score
        pattern_score = min(0.3, pattern_score)
        
        # Check emergent terminology
        emergent_matches = []
        for term in self.emergent_terms:
            term_lower = term.lower()
            if term_lower in text_lower:
                emergent_matches.append(term)
                # Update effectiveness tracking
                self._update_term_effectiveness(term, "emergent")
                
        # Calculate score based on emergent matches
        if emergent_matches:
            # Lower weight for emergent terms (less validated)
            emergent_score = min(0.4, len(emergent_matches) * 0.1)
        
        # Combine scores from all layers
        combined_score = core_score + pattern_score + emergent_score
        
        # Apply platform-specific adjustments
        platform_factor = data_collection_config.PLATFORM_RELEVANCE_FACTORS.get(platform, 1.0)
        
        # Cap at 1.0
        final_score = min(1.0, combined_score * platform_factor)
        
        return final_score
    
    def _update_term_effectiveness(self, term, term_type):
        """
        Update effectiveness tracking for a term.
        
        Args:
            term: The matched term
            term_type: Type of the term (core, pattern, emergent)
        """
        if term not in self.term_effectiveness:
            self.term_effectiveness[term] = {
                "type": term_type,
                "matches": 0,
                "last_match": None
            }
        
        self.term_effectiveness[term]["matches"] += 1
        self.term_effectiveness[term]["last_match"] = datetime.now()
    
    def evolve_lexicon(self, new_candidate_terms):
        """
        Evolve the lexicon with new candidate terms.
        
        Args:
            new_candidate_terms: Dictionary of term -> confidence_score
            
        Returns:
            list: Newly added terms
        """
        logger.info(f"Evolving lexicon with {len(new_candidate_terms)} candidate terms")
        
        added_terms = []
        
        for term, confidence in new_candidate_terms.items():
            # Skip terms already in any lexicon
            if term.lower() in (t.lower() for t in self.core_terms) or \
               term.lower() in (t.lower() for t in self.emergent_terms):
                continue
            
            # Add terms with sufficient confidence to emergent lexicon
            if confidence >= data_collection_config.LEXICON["emergence_threshold"]:
                self.emergent_terms.add(term)
                added_terms.append(term)
                logger.info(f"Added new term to emergent lexicon: {term}")
        
        # Update last update timestamp
        self.last_update = datetime.now()
        
        # Persist updated lexicon if terms were added
        if added_terms:
            self._persist_lexicon()
            
        return added_terms
    
    def _persist_lexicon(self):
        """Persist the current lexicon to storage."""
        try:
            lexicon_data = {
                "core_terms": list(self.core_terms),
                "contextual_patterns": self.contextual_patterns,
                "emergent_terms": list(self.emergent_terms),
                "last_update": self.last_update.isoformat()
            }
            
            with open("Data_Collection/lexicon.json", "w") as f:
                json.dump(lexicon_data, f, indent=2)
                
            logger.info("Successfully persisted updated lexicon")
            
        except Exception as e:
            logger.error(f"Error persisting lexicon: {str(e)}")
    
    def prune_lexicon(self):
        """
        Prune ineffective terms from the emergent lexicon.
        
        Returns:
            list: Removed terms
        """
        removed_terms = []
        
        # Only prune emergent terms, not core terminology
        for term in list(self.emergent_terms):
            if term in self.term_effectiveness:
                effectiveness = self.term_effectiveness[term]
                
                # Check if term hasn't matched in a while
                last_match = effectiveness.get("last_match")
                if last_match:
                    days_since_match = (datetime.now() - last_match).days
                    
                    # If term hasn't matched in 30 days and has few matches, remove it
                    if days_since_match > 30 and effectiveness.get("matches", 0) < 5:
                        self.emergent_terms.remove(term)
                        removed_terms.append(term)
                        logger.info(f"Pruned ineffective term from emergent lexicon: {term}")
        
        # Persist updated lexicon if terms were removed
        if removed_terms:
            self._persist_lexicon()
            
        return removed_terms
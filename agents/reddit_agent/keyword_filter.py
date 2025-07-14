# keyword_filter.py
import re
import json
import logging
from collections import Counter
from agents.reddit_agent.reddit_config import KEYWORD_LEXICON
from agents.reddit_agent.llm_analyzer import LLMAnalyzer

logger = logging.getLogger(__name__)

class KeywordFilter:
    """Filter content based on keyword relevance"""
    
    def __init__(self, lexicon=None):
        """
        Initialize the keyword filter
        
        Args:
            lexicon (list, optional): List of keywords to use for filtering
        """
        
        self.lexicon = lexicon or KEYWORD_LEXICON
        self.patterns = [re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE) for kw in self.lexicon]
        self.core_terms = set(kw.lower() for kw in self.lexicon)
        self.emergent_terms = set()
        self.llm = LLMAnalyzer()
        self.load_lexicon()
        # Create case-insensitive regex pattern for each keyword
        self.patterns = [re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE) 
                         for kw in self.lexicon]
        logger.info(f"Keyword filter initialized with {len(self.lexicon)} terms")
        
    def calculate_relevance(self, text):
        """
        Calculate relevance score based on keyword presence
        
        Args:
            text (str): Text content to analyze
            
        Returns:
            float: Relevance score between 0 and 1
        """
        if not text:
            return 0.0
        if len(text.split()) < 15:
            return 0.0

        text_lower = text.lower()

        # Strong compound term heuristics (run early — full override)
        if ("system prompt" in text_lower and "leak" in text_lower) or \
        ("base prompt" in text_lower and "dump" in text_lower) or \
        ("grok" in text_lower and "prompt" in text_lower):
            return 0.9  # strong override

        # Count keyword matches
        matches = Counter()
        for pattern in self.patterns:
            matches[pattern.pattern] = len(pattern.findall(text))

        total_matches = sum(matches.values())
        unique_matches = len([k for k, v in matches.items() if v > 0])

        if total_matches == 0:
            return 0.0

        # Custom weights
        diversity_weight = 0.6
        volume_weight = 0.3
        compound_weight = 0.1

        # Extra pattern bonus (soft boost)
        compound_bonus = 0.1 if ("prompt" in text_lower and "leak" in text_lower) else 0.0

        relevance = (
            diversity_weight * (unique_matches / len(self.patterns)) +
            volume_weight * min(1.0, total_matches / 10) +
            compound_bonus
        )

        # Step 3 fix: post-processing boost
        if "prompt" in text_lower and "chatgpt" in text_lower:
            relevance = max(0.5, relevance)

        return min(1.0, relevance)

    
    def update_lexicon(self, new_terms):
        """
        Update the keyword lexicon with new terms
        
        Args:
            new_terms (list): New terms to add to the lexicon
        """
        # Add only terms that don't already exist
        added_terms = []
        for term in new_terms:
            term = term.lower().strip()
            if term and term not in self.lexicon:
                self.lexicon.append(term)
                self.patterns.append(re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE))
                added_terms.append(term)
                
        if added_terms:
            logger.info(f"Added {len(added_terms)} new terms to lexicon: {', '.join(added_terms)}")
            
        return added_terms
    
    def expand_lexicon(self, examples: list[str]):
        """
        Ask the LLM to suggest new jailbreak keywords from sample prompts.
        """
        prompt = (
            "Here are example jailbreak prompts:\n\n"
            + "\n---\n".join(examples)
            + "\n\nSuggest 5 new single-phrase keywords that indicate LLM jailbreak attempts:"
        )
        suggestions = self.llm.complete_simple(prompt)
        # assume it returns a newline list
        for line in suggestions.splitlines():
            term = line.strip().lower()
            if term and term not in self.core_terms:
                self.emergent_terms.add(term)
                
    def save_lexicon(self, filepath="Data/Reddit_Data/lexicon.json"):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(sorted(set(self.lexicon)), f, ensure_ascii=False, indent=2)
        logger.info(f" Lexicon saved to {filepath}")

    def load_lexicon(self, filepath="Data/Reddit_Data/lexicon.json"):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.update_lexicon(loaded)
            logger.info(f" Lexicon loaded from {filepath}")
        except FileNotFoundError:
            logger.warning(f"No lexicon file found at {filepath} — using default")

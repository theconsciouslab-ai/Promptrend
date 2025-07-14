# keyword_filter.py
import re
import logging
from collections import Counter, defaultdict
import math
from agents.twitter_agent.twitter_config import KEYWORD_LEXICON

logger = logging.getLogger(__name__)

class KeywordFilter:
    """Filter content based on keyword relevance and manage keyword lexicon"""
    
    def __init__(self, lexicon=None):
        """
        Initialize the keyword filter
        
        Args:
            lexicon (list, optional): List of keywords to use for filtering
        """
        self.lexicon = lexicon or KEYWORD_LEXICON
        
        # Create regex patterns for each keyword
        self.patterns = [re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE) 
                         for kw in self.lexicon]
                         
        # Term frequency tracking across all documents
        self.term_frequencies = Counter()
        
        # Document frequency tracking (how many documents contain each term)
        self.document_frequency = Counter()
        
        # Total documents processed
        self.total_documents = 0
        
        # Co-occurrence matrix
        self.co_occurrences = defaultdict(Counter)
        
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
            
        # Count matches for each keyword
        matches = Counter()
        text_lower = text.lower()
        
        for idx, pattern in enumerate(self.patterns):
            keyword = self.lexicon[idx]
            matches[keyword] = len(pattern.findall(text))
            
        # Calculate weighted score
        total_matches = sum(matches.values())
        unique_matches = len([k for k, v in matches.items() if v > 0])
        
        # No matches
        if total_matches == 0:
            return 0.0
            
        # Calculate relevance based on number and diversity of matches
        # Weight uniqueness more than repetition
        relevance = min(1.0, (0.7 * (unique_matches / len(self.lexicon)) + 
                              0.3 * min(1.0, total_matches / 15)))
        
        return relevance
    
    def update_statistics(self, documents):
        """
        Update term statistics based on new documents
        
        Args:
            documents (list): List of text documents
        """
        for doc in documents:
            if not doc:
                continue
                
            # Track document frequency
            self.total_documents += 1
            
            # Process document
            doc_lower = doc.lower()
            
            # Count term frequencies in this document
            doc_terms = Counter()
            
            for idx, pattern in enumerate(self.patterns):
                keyword = self.lexicon[idx]
                occurrences = pattern.findall(doc_lower)
                
                if occurrences:
                    self.document_frequency[keyword] += 1
                    doc_terms[keyword] = len(occurrences)
                    
            # Update global term frequencies
            self.term_frequencies.update(doc_terms)
            
            # Update co-occurrence matrix
            term_list = list(doc_terms.keys())
            for i, term1 in enumerate(term_list):
                for term2 in term_list[i+1:]:
                    self.co_occurrences[term1][term2] += 1
                    self.co_occurrences[term2][term1] += 1
    
    def generate_new_terms(self, threshold=3):
        """
        Generate potential new terms based on co-occurrence
        
        Args:
            threshold (int): Minimum co-occurrence count to consider
            
        Returns:
            list: List of potential new terms
        """
        # Find strongly co-occurring terms
        candidates = []
        
        for term1, co_terms in self.co_occurrences.items():
            for term2, count in co_terms.items():
                if count >= threshold:
                    # Generate n-gram suggestion
                    candidate = f"{term1} {term2}"
                    candidates.append((candidate, count))
        
        # Sort by co-occurrence count
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Return top candidates
        return [term for term, count in candidates[:20]]
    
    def update_lexicon(self, new_terms):
        """
        Update the keyword lexicon with new terms
        
        Args:
            new_terms (list): New terms to add to the lexicon
            
        Returns:
            list: List of added terms
        """
        # Add only terms that don't already exist
        added_terms = []
        for term in new_terms:
            term = term.lower().strip()
            if term and term not in self.lexicon:
                self.lexicon.append(term)
                # Add new pattern
                self.patterns.append(re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE))
                added_terms.append(term)
                
        if added_terms:
            logger.info(f"Added {len(added_terms)} new terms to lexicon: {', '.join(added_terms)}")
            
        return added_terms
    
    def get_important_terms(self, top_n=10):
        """
        Get most important terms based on TF-IDF
        
        Args:
            top_n (int): Number of top terms to return
            
        Returns:
            list: List of (term, score) tuples
        """
        if self.total_documents == 0:
            return []
            
        # Calculate TF-IDF for each term
        tfidf_scores = {}
        
        for term, freq in self.term_frequencies.items():
            # Term frequency
            tf = freq
            
            # Inverse document frequency
            idf = math.log(self.total_documents / max(1, self.document_frequency[term]))
            
            # TF-IDF score
            tfidf_scores[term] = tf * idf
            
        # Sort by score
        top_terms = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
        
        return top_terms[:top_n]
    
    def generate_search_queries(self, max_queries=5, terms_per_query=3):
        """
        Generate optimized search queries based on important terms
        
        Args:
            max_queries (int): Maximum number of queries to generate
            terms_per_query (int): Number of terms per query
            
        Returns:
            list: List of search query strings
        """
        top_terms = self.get_important_terms(top_n=terms_per_query * max_queries)
        
        if not top_terms:
            return []
            
        # Group terms into queries
        queries = []
        for i in range(0, min(len(top_terms), max_queries * terms_per_query), terms_per_query):
            query_terms = [term for term, score in top_terms[i:i+terms_per_query]]
            query = " OR ".join(f'"{term}"' for term in query_terms)
            queries.append(query)
            
        return queries
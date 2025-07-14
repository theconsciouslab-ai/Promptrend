# Twitter/X Agent Implementation Guide

## Overview

This document provides a comprehensive guide for implementing a Twitter/X Agent that monitors LLM security-related discussions using graph centrality analysis and LLM-based content evaluation. The agent implements a dual-stream approach that combines user-focused monitoring with keyword-based searches to efficiently detect vulnerability discussions.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Implementation Requirements](#implementation-requirements)
3. [Core Components](#core-components)
4. [Implementation Steps](#implementation-steps)
5. [Code Implementation](#code-implementation)
6. [Deployment Considerations](#deployment-considerations)
7. [Examples and Usage](#examples-and-usage)

## Architecture Overview

The Twitter/X Agent follows a dual-stream architecture:

1. **Primary Stream (User-Focused Monitoring)**: Uses graph-theoretic centrality measures to identify and monitor influential security researchers and AI practitioners.

2. **Secondary Stream (Keyword-Based Discovery)**: Complements the primary stream by searching for relevant discussions from users outside the primary monitoring set using an evolving keyword lexicon.

Both streams converge at the content analysis stage, where an LLM evaluates the security relevance of collected content.

```
PRIMARY STREAM                        SECONDARY STREAM
─────────────────                     ─────────────────
Security Researchers                  Vulnerability Dataset
        │                                     │
        ▼                                     ▼
Graph Centrality Analysis             Lexicon Evolution
        │                                     │
        ▼                                     ▼
Prioritized User List                Enhanced Keyword Lexicon
        │                                     │
        ▼                                     ▼
Timeline & Reply Collection          API-Based Twitter Searches
        │                                     │
        └────────────────┬─────────────────────┘
                         │
                         ▼
              LLM-Based Content Analysis
                         │
                         ▼
              Vulnerability Repository
```

## Implementation Requirements

### Dependencies

- Python 3.9+
- Tweepy (Twitter API client)
- NetworkX (Graph analysis)
- OpenAI API, Anthropic API, or other LLM provider
- JSON storage/database system

### API Keys & Authentication

- Twitter/X API credentials (API key, API key secret, access token, access token secret)
- LLM provider API key (OpenAI, Anthropic, etc.)

## Core Components

### 1. Graph Centrality Analysis Module

Responsible for:
- Building a graph representation of security researchers and their connections
- Calculating centrality measures to identify influential users
- Generating a prioritized list of users to monitor

### 2. Twitter Data Collection Module

Responsible for:
- Collecting tweets from prioritized users
- Retrieving conversation threads (parent tweets and replies)
- Implementing rate limiting and error handling

### 3. Keyword Filtering Module

Responsible for:
- Pre-filtering collected content using keyword matching
- Maintaining and updating the keyword lexicon
- Generating optimized search queries

### 4. LLM Analysis Module

Responsible for:
- Preparing conversation context for LLM analysis
- Interfacing with LLM API
- Scoring content across multiple dimensions
- Filtering high-relevance content

### 5. Storage Module

Responsible for:
- Storing collected tweets and conversation threads
- Organizing vulnerability findings
- Managing the evolving keyword lexicon

## Implementation Steps

### Step 1: Set Up Environment

1. Create a virtual environment
2. Install required dependencies
3. Set up configuration for API keys and parameters

### Step 2: Implement Graph Centrality Analysis

1. Build user graph from seed researcher accounts
2. Calculate centrality measures (degree, eigenvector, PageRank)
3. Generate ranked user list

### Step 3: Implement Twitter Data Collection

1. Configure Tweepy client
2. Implement timeline and reply collection
3. Implement conversation thread reconstruction

### Step 4: Implement Keyword Filtering

1. Build the initial keyword lexicon
2. Implement filtering logic
3. Develop lexicon evolution mechanisms

### Step 5: Implement LLM Analysis

1. Set up connection to LLM API
2. Design prompts for vulnerability analysis
3. Implement multi-dimensional scoring system

### Step 6: Implement Storage

1. Set up JSON storage structure
2. Implement storage and retrieval functions
3. Create index for efficient querying

### Step 7: Integrate Components

1. Connect all modules in pipeline
2. Implement error handling and logging
3. Add monitoring and metrics

## Code Implementation

### Configuration Setup

```python
# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Twitter API Configuration
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")

# LLM API Configuration
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4-turbo")  # Default to GPT-4 Turbo
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # Options: openai, anthropic, etc.

# Agent Configuration
# Seed users (security researchers and AI practitioners to start graph analysis)
SEED_USERS = [
    "cybersecurityAI",
    "llm_security",
    "AIresearcher",
    "ethicalAIhacker",
    "promptsecurity",
    "LLMvulns",
    "jailbreakAI",
    "AIalignment",
    "securityLLM",
    "redteamAI"
]

# Collection parameters
USER_TWEETS_LIMIT = int(os.getenv("USER_TWEETS_LIMIT", 100))  # Number of tweets to collect per user
MAX_USERS = int(os.getenv("MAX_USERS", 2000))  # Maximum number of users to monitor
CONVERSATION_DEPTH = int(os.getenv("CONVERSATION_DEPTH", 5))  # Max depth for conversation reconstruction
COLLECTION_INTERVAL = int(os.getenv("COLLECTION_INTERVAL", 3600))  # Collection frequency in seconds

# Filtering parameters
KEYWORD_RELEVANCE_THRESHOLD = float(os.getenv("KEYWORD_RELEVANCE_THRESHOLD", 0.3))
LLM_RELEVANCE_THRESHOLD = float(os.getenv("LLM_RELEVANCE_THRESHOLD", 0.6))

# Initial keyword lexicon
KEYWORD_LEXICON = [
    "jailbreak", "DAN", "vulnerability", "bypass", "prompt injection",
    "safety", "alignment", "exploit", "hack", "red team", "red teaming",
    "security", "circumvent", "workaround", "backdoor", "attack",
    "hallucination", "extraction", "system prompt", "leaking",
    "RLHF", "data poisoning", "model poisoning", "adversarial",
    "sandbox", "escape", "authentication", "unauthorized access"
]

# Storage Configuration
DATA_DIR = os.getenv("DATA_DIR", "data")
```

### Graph Analysis Module

```python
# graph_analyzer.py
import logging
import networkx as nx
import tweepy
from collections import defaultdict
from config import TWITTER_BEARER_TOKEN, SEED_USERS, MAX_USERS

# Configure logging
logger = logging.getLogger(__name__)

class GraphAnalyzer:
    """
    Analyzes Twitter user relationships using graph centrality measures
    to identify influential security researchers and AI practitioners.
    """
    
    def __init__(self, bearer_token=None):
        """
        Initialize the graph analyzer with Twitter API credentials
        
        Args:
            bearer_token (str, optional): Twitter API bearer token
        """
        self.bearer_token = bearer_token or TWITTER_BEARER_TOKEN
        
        # Initialize Tweepy client
        self.client = tweepy.Client(bearer_token=self.bearer_token)
        
        # Initialize graph
        self.graph = nx.DiGraph()
        
        logger.info("Graph analyzer initialized")
        
    def build_graph(self, seed_users=None, max_users=MAX_USERS, depth=2):
        """
        Build a directed graph of Twitter users based on following relationships
        
        Args:
            seed_users (list, optional): Initial set of users to start graph building
            max_users (int, optional): Maximum number of users to include in graph
            depth (int, optional): Depth of traversal from seed users
            
        Returns:
            nx.DiGraph: The constructed graph
        """
        logger.info(f"Building graph from {len(seed_users or SEED_USERS)} seed users")
        
        seed_users = seed_users or SEED_USERS
        visited = set()
        to_visit = [(user, 0) for user in seed_users]  # (username, depth)
        
        # Add seed users to graph
        for user in seed_users:
            self.graph.add_node(user, seed=True)
        
        # Breadth-first traversal
        while to_visit and len(self.graph) < max_users:
            current_user, current_depth = to_visit.pop(0)
            
            if current_user in visited:
                continue
                
            visited.add(current_user)
            
            try:
                # Get user details
                user_info = self.client.get_user(username=current_user)
                if not user_info.data:
                    continue
                    
                user_id = user_info.data.id
                
                # Get following relationships (users this account follows)
                following = []
                for response in tweepy.Paginator(
                    self.client.get_users_following,
                    id=user_id,
                    max_results=100,
                    limit=5  # Limit API calls
                ):
                    if response.data:
                        following.extend([user.username for user in response.data])
                
                # Add edges to graph
                for followed_user in following:
                    if followed_user not in self.graph:
                        self.graph.add_node(followed_user)
                    
                    # Edge direction: current_user -> followed_user
                    self.graph.add_edge(current_user, followed_user)
                
                # Add users to visit queue if within depth limit
                if current_depth < depth:
                    for followed_user in following:
                        if followed_user not in visited:
                            to_visit.append((followed_user, current_depth + 1))
                
                logger.info(f"Processed user {current_user}: {len(following)} following")
                
            except Exception as e:
                logger.error(f"Error processing user {current_user}: {str(e)}")
            
        logger.info(f"Graph building completed: {len(self.graph)} users, {self.graph.number_of_edges()} connections")
        return self.graph
        
    def calculate_centrality(self):
        """
        Calculate various centrality measures for the graph
        
        Returns:
            dict: Dictionary mapping usernames to combined centrality scores
        """
        if not self.graph or self.graph.number_of_nodes() == 0:
            logger.warning("Graph is empty, cannot calculate centrality")
            return {}
            
        logger.info("Calculating centrality measures")
        
        # Calculate different centrality measures
        degree_centrality = nx.degree_centrality(self.graph)
        in_degree_centrality = nx.in_degree_centrality(self.graph)
        eigenvector_centrality = nx.eigenvector_centrality_numpy(self.graph, weight=None)
        
        try:
            pagerank = nx.pagerank(self.graph, alpha=0.85)
        except:
            logger.warning("PageRank calculation failed, using default values")
            pagerank = {node: 0.0 for node in self.graph.nodes()}
        
        # Combine centrality scores with weights
        combined_scores = {}
        
        for user in self.graph.nodes():
            # Weight each centrality measure
            combined_scores[user] = (
                0.2 * degree_centrality.get(user, 0) +
                0.3 * in_degree_centrality.get(user, 0) +
                0.2 * eigenvector_centrality.get(user, 0) +
                0.3 * pagerank.get(user, 0)
            )
            
            # Boost scores for seed users
            if self.graph.nodes[user].get('seed', False):
                combined_scores[user] *= 1.2
                
        logger.info("Centrality calculation completed")
        return combined_scores
        
    def get_top_users(self, limit=MAX_USERS):
        """
        Get top users ranked by centrality
        
        Args:
            limit (int): Maximum number of users to return
            
        Returns:
            list: List of (username, score) tuples sorted by score
        """
        combined_scores = self.calculate_centrality()
        
        # Sort users by combined score
        ranked_users = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        logger.info(f"Generated ranked list of {len(ranked_users)} users")
        return ranked_users[:limit]
```

### Twitter Client Module

```python
# twitter_client.py
import logging
import tweepy
import time
from config import (
    TWITTER_API_KEY, TWITTER_API_SECRET,
    TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET,
    TWITTER_BEARER_TOKEN, USER_TWEETS_LIMIT,
    CONVERSATION_DEPTH
)

# Configure logging
logger = logging.getLogger(__name__)

class TwitterClient:
    """Client for collecting tweets and conversation threads from Twitter/X API"""
    
    def __init__(self, bearer_token=None, api_key=None, api_secret=None, 
                 access_token=None, access_secret=None):
        """
        Initialize the Twitter client with API credentials
        
        Args:
            bearer_token (str, optional): Twitter API bearer token
            api_key (str, optional): Twitter API key
            api_secret (str, optional): Twitter API secret
            access_token (str, optional): Twitter access token
            access_secret (str, optional): Twitter access token secret
        """
        self.bearer_token = bearer_token or TWITTER_BEARER_TOKEN
        self.api_key = api_key or TWITTER_API_KEY
        self.api_secret = api_secret or TWITTER_API_SECRET
        self.access_token = access_token or TWITTER_ACCESS_TOKEN
        self.access_secret = access_secret or TWITTER_ACCESS_SECRET
        
        # Initialize Tweepy clients
        self.client = tweepy.Client(
            bearer_token=self.bearer_token,
            consumer_key=self.api_key,
            consumer_secret=self.api_secret,
            access_token=self.access_token,
            access_token_secret=self.access_secret
        )
        
        logger.info("Twitter client initialized")
        
    def get_user_tweets(self, username, max_count=USER_TWEETS_LIMIT, include_replies=True):
        """
        Get recent tweets from a specific user
        
        Args:
            username (str): Twitter username
            max_count (int, optional): Maximum number of tweets to retrieve
            include_replies (bool, optional): Whether to include replies in results
            
        Returns:
            list: List of tweet objects
        """
        try:
            # Get user ID from username
            user = self.client.get_user(username=username)
            if not user.data:
                logger.warning(f"User {username} not found")
                return []
                
            user_id = user.data.id
            
            # Define tweet fields to retrieve
            tweet_fields = [
                'id', 'text', 'author_id', 'created_at', 'conversation_id',
                'public_metrics', 'referenced_tweets', 'in_reply_to_user_id'
            ]
            
            # Collect tweets
            tweets = []
            
            # Get user timeline
            for response in tweepy.Paginator(
                self.client.get_users_tweets,
                id=user_id,
                max_results=100,  # Maximum allowed per request
                tweet_fields=tweet_fields,
                exclude=['retweets'] if include_replies else ['retweets', 'replies'],
                limit=(max_count // 100) + 1
            ):
                if response.data:
                    tweets.extend(response.data)
                    if len(tweets) >= max_count:
                        tweets = tweets[:max_count]
                        break
                        
            logger.info(f"Collected {len(tweets)} tweets from user {username}")
            return tweets
            
        except Exception as e:
            logger.error(f"Error collecting tweets from {username}: {str(e)}")
            return []
            
    def search_tweets(self, query, max_count=100):
        """
        Search for tweets matching a specific query
        
        Args:
            query (str): Search query
            max_count (int, optional): Maximum number of tweets to retrieve
            
        Returns:
            list: List of tweet objects
        """
        try:
            # Define tweet fields to retrieve
            tweet_fields = [
                'id', 'text', 'author_id', 'created_at', 'conversation_id',
                'public_metrics', 'referenced_tweets', 'in_reply_to_user_id'
            ]
            
            # Collect tweets
            tweets = []
            
            # Search for tweets (recent search only for most non-academic accounts)
            for response in tweepy.Paginator(
                self.client.search_recent_tweets,
                query=query,
                max_results=100,  # Maximum allowed per request
                tweet_fields=tweet_fields,
                limit=(max_count // 100) + 1
            ):
                if response.data:
                    tweets.extend(response.data)
                    if len(tweets) >= max_count:
                        tweets = tweets[:max_count]
                        break
                        
            logger.info(f"Search for '{query}' returned {len(tweets)} results")
            return tweets
            
        except Exception as e:
            logger.error(f"Error searching for '{query}': {str(e)}")
            return []
            
    def reconstruct_conversation(self, tweet, max_depth=CONVERSATION_DEPTH):
        """
        Reconstruct a conversation thread around a tweet
        
        Args:
            tweet: Tweet object
            max_depth (int, optional): Maximum depth of conversation to retrieve
            
        Returns:
            dict: Conversation thread organized by tweet ID
        """
        if not tweet:
            return {}
            
        conversation = {tweet.id: tweet}
        
        try:
            # First, get parent tweets (traverse upward)
            current_tweet = tweet
            depth = 0
            
            while depth < max_depth:
                # Check if tweet is a reply
                if not hasattr(current_tweet, 'referenced_tweets'):
                    break
                    
                referenced_tweets = current_tweet.referenced_tweets
                
                # Find parent tweet reference
                parent_id = None
                if referenced_tweets:
                    for ref in referenced_tweets:
                        if ref.type == 'replied_to':
                            parent_id = ref.id
                            break
                
                if not parent_id:
                    break
                    
                # Get parent tweet
                parent_response = self.client.get_tweet(
                    parent_id,
                    tweet_fields=['id', 'text', 'author_id', 'created_at', 
                                 'conversation_id', 'referenced_tweets']
                )
                
                if not parent_response.data:
                    break
                    
                parent_tweet = parent_response.data
                conversation[parent_tweet.id] = parent_tweet
                
                # Continue traversal with parent
                current_tweet = parent_tweet
                depth += 1
                
                # Rate limiting - avoid hitting limits
                time.sleep(0.2)
            
            # Then, get replies (traverse downward)
            # Note: This is limited in Twitter API v2 without Academic Access
            if hasattr(tweet, 'conversation_id'):
                conversation_id = tweet.conversation_id
                
                # Search for tweets in the same conversation
                search_query = f"conversation_id:{conversation_id}"
                
                replies = self.search_tweets(search_query, max_count=20)
                
                for reply in replies:
                    if reply.id not in conversation:
                        conversation[reply.id] = reply
            
            logger.info(f"Reconstructed conversation with {len(conversation)} tweets")
            return conversation
            
        except Exception as e:
            logger.error(f"Error reconstructing conversation: {str(e)}")
            return {tweet.id: tweet}  # Return at least the original tweet
```

### Keyword Filtering Module

```python
# keyword_filter.py
import re
import logging
from collections import Counter, defaultdict
import math
from config import KEYWORD_LEXICON

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
```

### LLM Analysis Module

```python
# llm_analyzer.py
import logging
import json
import time
import openai
import anthropic
from config import LLM_API_KEY, LLM_MODEL, LLM_PROVIDER

logger = logging.getLogger(__name__)

class LLMAnalyzer:
    """Analyze content using LLM-based evaluation"""
    
    def __init__(self):
        """Initialize the LLM analyzer with API configuration"""
        self.provider = LLM_PROVIDER
        self.model = LLM_MODEL
        
        # Configure API client based on provider
        if self.provider == "openai":
            openai.api_key = LLM_API_KEY
            self.client = openai.OpenAI()
        elif self.provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=LLM_API_KEY)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
            
        logger.info(f"LLM analyzer initialized with {self.provider} ({self.model})")
        
    def analyze_content(self, tweet_data, conversation):
        """
        Analyze tweet and conversation for LLM security relevance
        
        Args:
            tweet_data (dict): Tweet data including id and text
            conversation (dict): Conversation thread data
            
        Returns:
            dict: Analysis results with scores and metadata
        """
        # Prepare content for analysis
        content = self._prepare_conversation_context(tweet_data, conversation)
        
        # Run analyses in sequence
        technical_score = self._analyze_technical_relevance(content)
        security_score = self._analyze_security_impact(content)
        llm_specific_score = self._analyze_llm_specific(content)
        
        # Calculate combined score
        combined_score = (0.4 * technical_score + 
                          0.4 * security_score + 
                          0.2 * llm_specific_score)
        
        # Extract key insights from content
        insights = self._extract_key_insights(content)
        
        return {
            "scores": {
                "technical": technical_score,
                "security": security_score,
                "llm_specific": llm_specific_score,
                "combined": combined_score
            },
            "insights": insights,
            "timestamp": time.time()
        }
        
    def _prepare_conversation_context(self, primary_tweet, conversation):
        """
        Prepare conversation context for LLM analysis
        
        Args:
            primary_tweet (dict): Primary tweet data
            conversation (dict): Conversation thread data
            
        Returns:
            str: Formatted content for analysis
        """
        # Get tweet ID
        tweet_id = primary_tweet.get('id') or primary_tweet.id
        
        # Create dictionary of all tweets
        all_tweets = {}
        
        # Add main tweet to dictionary
        if isinstance(primary_tweet, dict):
            all_tweets[tweet_id] = primary_tweet
        else:
            # Convert tweet object to dict
            all_tweets[tweet_id] = {
                'id': primary_tweet.id,
                'text': primary_tweet.text,
                'author_id': primary_tweet.author_id,
                'created_at': primary_tweet.created_at
            }
        
        # Add conversation tweets
        for tweet_id, tweet in conversation.items():
            if isinstance(tweet, dict):
                all_tweets[tweet_id] = tweet
            else:
                # Convert tweet object to dict
                all_tweets[tweet_id] = {
                    'id': tweet.id,
                    'text': tweet.text,
                    'author_id': tweet.author_id,
                    'created_at': tweet.created_at
                }
        
        # Organize tweets by reference chain
        # Start with primary tweet
        ordered_tweets = [all_tweets[tweet_id]]
        
        # Find parent tweets (tweets that the primary tweet replies to)
        parent_ids = self._find_parent_ids(primary_tweet)
        parent_tweets = [all_tweets.get(pid) for pid in parent_ids if pid in all_tweets]
        parent_tweets.reverse()  # Oldest first
        
        # Add parent tweets to beginning
        ordered_tweets = parent_tweets + ordered_tweets
        
        # Find child tweets (replies to the primary tweet)
        child_tweets = []
        for tid, tweet in all_tweets.items():
            if tid != tweet_id and tid not in parent_ids:
                if 'in_reply_to_status_id' in tweet and tweet['in_reply_to_status_id'] == tweet_id:
                    child_tweets.append(tweet)
                elif hasattr(tweet, 'referenced_tweets'):
                    for ref in tweet.referenced_tweets or []:
                        if ref.type == 'replied_to' and ref.id == tweet_id:
                            child_tweets.append(tweet)
                            break
        
        # Sort child tweets by timestamp if available
        child_tweets.sort(key=lambda t: t.get('created_at', 0))
        
        # Add child tweets to end
        ordered_tweets.extend(child_tweets)
        
        # Format conversation
        content = "CONVERSATION THREAD:\n\n"
        
        for i, tweet in enumerate(ordered_tweets):
            # Get author info
            author_id = tweet.get('author_id') or getattr(tweet, 'author_id', 'unknown_user')
            
            # Get tweet text
            if isinstance(tweet, dict):
                text = tweet.get('text', '')
            else:
                text = tweet.text
            
            # Add to content
            content += f"[Tweet {i+1}] User {author_id}: {text}\n\n"
            
        return content
    
    def _find_parent_ids(self, tweet):
        """
        Find parent tweet IDs for a tweet
        
        Args:
            tweet: Tweet object or dictionary
            
        Returns:
            list: List of parent tweet IDs
        """
        parent_ids = []
        
        # Check if tweet is a dictionary
        if isinstance(tweet, dict):
            # Check for in_reply_to_status_id
            if 'in_reply_to_status_id' in tweet:
                parent_ids.append(tweet['in_reply_to_status_id'])
            # Check for referenced_tweets
            if 'referenced_tweets' in tweet:
                for ref in tweet['referenced_tweets'] or []:
                    if ref.get('type') == 'replied_to':
                        parent_ids.append(ref.get('id'))
        else:
            # Check for referenced_tweets attribute
            if hasattr(tweet, 'referenced_tweets'):
                for ref in tweet.referenced_tweets or []:
                    if ref.type == 'replied_to':
                        parent_ids.append(ref.id)
        
        return parent_ids
        
    def _analyze_technical_relevance(self, content):
        """
        Analyze content for technical relevance to LLM security
        
        Args:
            content (str): Content to analyze
            
        Returns:
            float: Technical relevance score (0-1)
        """
        prompt = """
        Analyze the following Twitter conversation and determine its technical relevance to LLM security.
        Focus on technical details, methods, code, or specific techniques related to:
        - LLM vulnerabilities
        - Jailbreak techniques
        - Prompt injection
        - LLM security measures
        - Technical bypass methods
        
        Conversation:
        
        [CONTENT]
        
        Score the technical relevance from 0.0 to 1.0, where:
        - 0.0: No technical LLM security content at all
        - 0.5: Some technical details but limited depth
        - 1.0: Highly technical, detailed LLM security discussion
        
        Return only a JSON object with the format: {"score": X.X, "reasoning": "brief explanation"}
        """
        
        prompt = prompt.replace("[CONTENT]", content)
        return self._get_llm_score(prompt)
        
    def _analyze_security_impact(self, content):
        """
        Analyze content for security impact and vulnerability presence
        
        Args:
            content (str): Content to analyze
            
        Returns:
            float: Security impact score (0-1)
        """
        prompt = """
        Analyze the following Twitter conversation and determine its relevance to LLM security vulnerabilities.
        Focus on security aspects such as:
        - Potential for harm or misuse
        - Security implications
        - Exploitability of described techniques
        - Effectiveness of described vulnerabilities
        - Risk level of disclosed methods
        
        Conversation:
        
        [CONTENT]
        
        Score the security impact from 0.0 to 1.0, where:
        - 0.0: No security implications or vulnerabilities
        - 0.5: Moderate security concerns or theoretical vulnerabilities
        - 1.0: Serious security vulnerability with demonstrated effectiveness
        
        Return only a JSON object with the format: {"score": X.X, "reasoning": "brief explanation"}
        """
        
        prompt = prompt.replace("[CONTENT]", content)
        return self._get_llm_score(prompt)
        
    def _analyze_llm_specific(self, content):
        """
        Analyze content for LLM-specific jailbreak/bypass techniques
        
        Args:
            content (str): Content to analyze
            
        Returns:
            float: LLM-specific relevance score (0-1)
        """
        prompt = """
        Analyze the following Twitter conversation and determine if it contains specific jailbreak or bypass techniques for LLMs.
        Focus on:
        - Named jailbreak methods (DAN, etc.)
        - Specific prompt templates
        - Model-specific vulnerabilities
        - Novel bypass approaches
        - Safety alignment circumvention
        
        Conversation:
        
        [CONTENT]
        
        Score the LLM-specific relevance from 0.0 to 1.0, where:
        - 0.0: No mention of specific LLM jailbreak/bypass techniques
        - 0.5: References to techniques but without specific details
        - 1.0: Detailed, specific jailbreak/bypass techniques with examples
        
        Return only a JSON object with the format: {"score": X.X, "reasoning": "brief explanation"}
        """
        
        prompt = prompt.replace("[CONTENT]", content)
        return self._get_llm_score(prompt)
        
    def _extract_key_insights(self, content):
        """
        Extract key insights about the vulnerability or technique discussed
        
        Args:
            content (str): Content to analyze
            
        Returns:
            dict: Extracted insights
        """
        prompt = """
        Analyze the following Twitter conversation about LLM security and extract key insights.
        
        Conversation:
        
        [CONTENT]
        
        Extract and return ONLY a JSON object with the following fields:
        {
            "vulnerability_type": "The type of vulnerability or technique discussed (if any)",
            "target_models": ["List of specific LLM models mentioned as targets"],
            "effectiveness": "Assessment of the reported effectiveness (if mentioned)",
            "novelty": "Whether this appears to be a novel technique or a known one",
            "key_techniques": ["List of key techniques or methods described"],
            "potential_mitigations": ["List of potential mitigations mentioned (if any)"]
        }
        
        If the conversation doesn't discuss LLM vulnerabilities or security, return null values.
        """
        
        prompt = prompt.replace("[CONTENT]", content)
        
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a security analyst specializing in LLM vulnerabilities."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    response_format={"type": "json_object"}
                )
                result = json.loads(response.choices[0].message.content)
                
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    temperature=0,
                    system="You are a security analyst specializing in LLM vulnerabilities. You extract insights in JSON format.",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                result = json.loads(response.content[0].text)
                
            return result
            
        except Exception as e:
            logger.error(f"Error extracting insights: {str(e)}")
            return {
                "vulnerability_type": None,
                "target_models": [],
                "effectiveness": None,
                "novelty": None,
                "key_techniques": [],
                "potential_mitigations": []
            }
    
    def _get_llm_score(self, prompt):
        """
        Get score from LLM based on prompt
        
        Args:
            prompt (str): Prompt to send to LLM
            
        Returns:
            float: Extracted score
        """
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a security analyst specializing in LLM vulnerabilities."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    response_format={"type": "json_object"}
                )
                result = json.loads(response.choices[0].message.content)
                
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    temperature=0,
                    system="You are a security analyst specializing in LLM vulnerabilities. You return scores in JSON format.",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                result = json.loads(response.content[0].text)
                
            # Extract score from result
            score = float(result.get("score", 0.0))
            
            # Ensure score is within valid range
            score = max(0.0, min(1.0, score))
            
            return score
            
        except Exception as e:
            logger.error(f"Error getting LLM score: {str(e)}")
            return 0.0
```

### JSON Storage Module

```python
# json_storage.py
import json
import os
import time
import logging
from pathlib import Path
from config import DATA_DIR

logger = logging.getLogger(__name__)

class JSONStorage:
    """Storage manager for saving Twitter data and analysis results as JSON files"""
    
    def __init__(self, data_dir=None):
        """
        Initialize the JSON storage
        
        Args:
            data_dir (str, optional): Directory to store JSON files
        """
        self.data_dir = data_dir or DATA_DIR
        
        # Create data directories if they don't exist
        self.tweets_dir = os.path.join(self.data_dir, "tweets")
        self.conversations_dir = os.path.join(self.data_dir, "conversations")
        self.analysis_dir = os.path.join(self.data_dir, "analysis")
        self.lexicon_dir = os.path.join(self.data_dir, "lexicon")
        self.index_file = os.path.join(self.data_dir, "index.json")
        
        Path(self.tweets_dir).mkdir(parents=True, exist_ok=True)
        Path(self.conversations_dir).mkdir(parents=True, exist_ok=True)
        Path(self.analysis_dir).mkdir(parents=True, exist_ok=True)
        Path(self.lexicon_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize index if it doesn't exist
        if not os.path.exists(self.index_file):
            self._write_json(self.index_file, {
                "tweets": {},
                "last_updated": int(time.time())
            })
            
        # Load current lexicon
        self.lexicon_file = os.path.join(self.lexicon_dir, "current_lexicon.json")
        if not os.path.exists(self.lexicon_file):
            from config import KEYWORD_LEXICON
            self._write_json(self.lexicon_file, {
                "terms": KEYWORD_LEXICON,
                "last_updated": int(time.time())
            })
            
        logger.info(f"JSON storage initialized at {self.data_dir}")
        
    def store_tweet(self, tweet):
        """
        Store a tweet as a JSON file
        
        Args:
            tweet: Tweet object or dictionary
            
        Returns:
            str: Tweet ID
        """
        try:
            # Extract tweet data
            if hasattr(tweet, 'id'):
                # Tweepy object
                tweet_id = str(tweet.id)
                
                # Convert tweet to dictionary
                data = {
                    "id": tweet_id,
                    "text": tweet.text,
                    "author_id": tweet.author_id,
                    "created_at": str(tweet.created_at),
                    "collected_at": int(time.time())
                }
                
                # Add additional fields if available
                if hasattr(tweet, 'conversation_id'):
                    data["conversation_id"] = tweet.conversation_id
                
                if hasattr(tweet, 'public_metrics'):
                    data["public_metrics"] = tweet.public_metrics._json
                
                if hasattr(tweet, 'referenced_tweets') and tweet.referenced_tweets:
                    data["referenced_tweets"] = [
                        {"type": ref.type, "id": ref.id} 
                        for ref in tweet.referenced_tweets
                    ]
                
            else:
                # Assume dictionary
                tweet_id = str(tweet.get('id'))
                data = tweet
            
            # Save tweet data
            tweet_file = os.path.join(self.tweets_dir, f"{tweet_id}.json")
            self._write_json(tweet_file, data)
            
            # Update index
            self._update_index(tweet_id, data)
            
            return tweet_id
            
        except Exception as e:
            logger.error(f"Error storing tweet: {str(e)}")
            return None
            
    def store_conversation(self, conversation_id, conversation):
        """
        Store a conversation thread as a JSON file
        
        Args:
            conversation_id (str): Conversation ID
            conversation (dict): Conversation data (dictionary of tweets)
            
        Returns:
            bool: Success status
        """
        try:
            # Convert conversation to dictionary format
            conversation_data = {
                "conversation_id": conversation_id,
                "collected_at": int(time.time()),
                "tweets": {}
            }
            
            for tweet_id, tweet in conversation.items():
                # Store individual tweets as well
                self.store_tweet(tweet)
                
                # Include in conversation
                if hasattr(tweet, 'id'):
                    # Tweepy object
                    tweet_data = {
                        "id": tweet.id,
                        "text": tweet.text,
                        "author_id": tweet.author_id,
                        "created_at": str(tweet.created_at)
                    }
                else:
                    # Assume dictionary
                    tweet_data = tweet
                
                conversation_data["tweets"][str(tweet_id)] = tweet_data
            
            # Save conversation data
            conversation_file = os.path.join(self.conversations_dir, f"{conversation_id}.json")
            self._write_json(conversation_file, conversation_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing conversation {conversation_id}: {str(e)}")
            return False
            
    def store_analysis(self, tweet_id, analysis_results):
        """
        Store analysis results as a JSON file
        
        Args:
            tweet_id (str): Tweet ID
            analysis_results (dict): Analysis results
            
        Returns:
            bool: Success status
        """
        try:
            # Add tweet_id to analysis results
            analysis_data = analysis_results.copy()
            analysis_data["tweet_id"] = tweet_id
            
            # Save analysis results
            analysis_file = os.path.join(self.analysis_dir, f"{tweet_id}.json")
            self._write_json(analysis_file, analysis_data)
            
            # Update index with score
            self._update_index_score(tweet_id, analysis_results.get("scores", {}).get("combined", 0.0))
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing analysis for tweet {tweet_id}: {str(e)}")
            return False
            
    def store_lexicon(self, lexicon):
        """
        Store current lexicon
        
        Args:
            lexicon (list): Keyword lexicon
            
        Returns:
            bool: Success status
        """
        try:
            # Create timestamped backup first
            current_time = int(time.time())
            backup_file = os.path.join(self.lexicon_dir, f"lexicon_{current_time}.json")
            
            if os.path.exists(self.lexicon_file):
                current_data = self._read_json(self.lexicon_file)
                self._write_json(backup_file, current_data)
            
            # Save new lexicon
            lexicon_data = {
                "terms": lexicon,
                "last_updated": current_time
            }
            self._write_json(self.lexicon_file, lexicon_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing lexicon: {str(e)}")
            return False
            
    def get_tweet(self, tweet_id):
        """
        Get tweet data from storage
        
        Args:
            tweet_id (str): Tweet ID
            
        Returns:
            dict: Tweet data or None if not found
        """
        tweet_file = os.path.join(self.tweets_dir, f"{tweet_id}.json")
        if os.path.exists(tweet_file):
            return self._read_json(tweet_file)
        return None
            
    def get_conversation(self, conversation_id):
        """
        Get conversation data from storage
        
        Args:
            conversation_id (str): Conversation ID
            
        Returns:
            dict: Conversation data or None if not found
        """
        conversation_file = os.path.join(self.conversations_dir, f"{conversation_id}.json")
        if os.path.exists(conversation_file):
            return self._read_json(conversation_file)
        return None
            
    def get_analysis(self, tweet_id):
        """
        Get analysis results from storage
        
        Args:
            tweet_id (str): Tweet ID
            
        Returns:
            dict: Analysis results or None if not found
        """
        analysis_file = os.path.join(self.analysis_dir, f"{tweet_id}.json")
        if os.path.exists(analysis_file):
            return self._read_json(analysis_file)
        return None
            
    def get_lexicon(self):
        """
        Get current lexicon
        
        Returns:
            list: Current lexicon
        """
        if os.path.exists(self.lexicon_file):
            data = self._read_json(self.lexicon_file)
            return data.get("terms", [])
        
        # Fallback to config
        from config import KEYWORD_LEXICON
        return KEYWORD_LEXICON
            
    def get_relevant_tweets(self, min_score=0.6, limit=50):
        """
        Get tweets with high relevance scores
        
        Args:
            min_score (float): Minimum combined score
            limit (int): Maximum number of tweets to return
            
        Returns:
            list: List of relevant tweets with analysis
        """
        try:
            # Read index
            index_data = self._read_json(self.index_file)
            tweets_index = index_data.get("tweets", {})
            
            # Filter tweets by score
            relevant_tweets = []
            
            for tweet_id, tweet_info in tweets_index.items():
                if tweet_info.get("score", 0.0) >= min_score:
                    # Get tweet data
                    tweet_data = self.get_tweet(tweet_id)
                    if tweet_data:
                        # Get analysis data
                        analysis_data = self.get_analysis(tweet_id)
                        
                        # Combine data
                        result = {
                            "tweet": tweet_data,
                            "analysis": analysis_data,
                            "relevance_score": tweet_info.get("score", 0.0)
                        }
                        
                        relevant_tweets.append(result)
            
            # Sort by score (descending)
            relevant_tweets.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            # Limit results
            return relevant_tweets[:limit]
            
        except Exception as e:
            logger.error(f"Error retrieving relevant tweets: {str(e)}")
            return []
    
    def _write_json(self, file_path, data):
        """
        Write data to a JSON file
        
        Args:
            file_path (str): Path to JSON file
            data (dict): Data to write
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    def _read_json(self, file_path):
        """
        Read data from a JSON file
        
        Args:
            file_path (str): Path to JSON file
            
        Returns:
            dict: JSON data
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    def _update_index(self, tweet_id, tweet_data):
        """
        Update the index with tweet information
        
        Args:
            tweet_id (str): Tweet ID
            tweet_data (dict): Tweet data
        """
        index_data = self._read_json(self.index_file)
        
        # Update tweet entry
        if tweet_id not in index_data["tweets"]:
            index_data["tweets"][tweet_id] = {}
            
        created_at = tweet_data.get("created_at")
        if isinstance(created_at, str):
            # Try to convert string to timestamp
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                created_at = dt.timestamp()
            except:
                created_at = 0
        
        index_data["tweets"][tweet_id].update({
            "author_id": tweet_data.get("author_id"),
            "created_at": created_at,
            "collected_at": tweet_data.get("collected_at", int(time.time()))
        })
        
        # Update last_updated timestamp
        index_data["last_updated"] = int(time.time())
        
        # Write updated index
        self._write_json(self.index_file, index_data)
        
    def _update_index_score(self, tweet_id, score):
        """
        Update the index with tweet score
        
        Args:
            tweet_id (str): Tweet ID
            score (float): Relevance score
        """
        index_data = self._read_json(self.index_file)
        
        # Update tweet score
        if tweet_id in index_data["tweets"]:
            index_data["tweets"][tweet_id]["score"] = score
            
        # Update last_updated timestamp
        index_data["last_updated"] = int(time.time())
        
        # Write updated index
        self._write_json(self.index_file, index_data)
```

### Main Agent Implementation

```python
# twitter_agent.py
import time
import logging
import threading
from queue import Queue

from config import (
    SEED_USERS, MAX_USERS, USER_TWEETS_LIMIT,
    COLLECTION_INTERVAL, CONVERSATION_DEPTH,
    KEYWORD_RELEVANCE_THRESHOLD, LLM_RELEVANCE_THRESHOLD,
    KEYWORD_LEXICON
)
from graph_analyzer import GraphAnalyzer
from twitter_client import TwitterClient
from keyword_filter import KeywordFilter
from llm_analyzer import LLMAnalyzer
from json_storage import JSONStorage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("twitter_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TwitterAgent:
    """
    Agent for monitoring Twitter/X discussions related to LLM security
    
    This agent implements a dual-stream approach that combines user-focused
    monitoring with keyword-based searches, using LLM-based content analysis
    to identify relevant vulnerability discussions.
    """
    
    def __init__(self):
        """Initialize the Twitter agent components"""
        self.graph_analyzer = GraphAnalyzer()
        self.twitter_client = TwitterClient()
        self.keyword_filter = KeywordFilter()
        self.llm_analyzer = LLMAnalyzer()
        self.storage = JSONStorage()
        self.running = False
        
        # Ranked list of users to monitor
        self.ranked_users = []
        
        # Processing queue for handling tweets asynchronously
        self.queue = Queue()
        
        # Statistics tracking
        self.stats = {
            "tweets_collected": 0,
            "tweets_analyzed": 0,
            "relevant_found": 0,
            "keywords_added": 0
        }
        
        logger.info("Twitter agent initialized")
        
    def start(self):
        """Start the agent's collection and processing threads"""
        if self.running:
            logger.warning("Agent is already running")
            return
            
        self.running = True
        
        # Start user ranking thread
        self.user_ranking_thread = threading.Thread(
            target=self._user_ranking_loop,
            daemon=True
        )
        self.user_ranking_thread.start()
        
        # Start user collection thread
        self.user_collection_thread = threading.Thread(
            target=self._user_collection_loop,
            daemon=True
        )
        self.user_collection_thread.start()
        
        # Start keyword search thread
        self.keyword_search_thread = threading.Thread(
            target=self._keyword_search_loop,
            daemon=True
        )
        self.keyword_search_thread.start()
        
        # Start worker threads for processing
        num_workers = 2  # Number of worker threads
        self.worker_threads = []
        
        for i in range(num_workers):
            worker = threading.Thread(
                target=self._processing_worker,
                daemon=True
            )
            worker.start()
            self.worker_threads.append(worker)
            
        logger.info(f"Agent started with {num_workers} worker threads")
        
    def stop(self):
        """Stop the agent's collection and processing"""
        if not self.running:
            return
            
        self.running = False
        
        # Wait for threads to finish
        for thread_name, thread in [
            ("user_ranking_thread", getattr(self, 'user_ranking_thread', None)),
            ("user_collection_thread", getattr(self, 'user_collection_thread', None)),
            ("keyword_search_thread", getattr(self, 'keyword_search_thread', None))
        ]:
            if thread:
                logger.info(f"Waiting for {thread_name} to finish...")
                thread.join(timeout=5)
            
        for i, thread in enumerate(getattr(self, 'worker_threads', [])):
            logger.info(f"Waiting for worker thread {i} to finish...")
            thread.join(timeout=5)
            
        logger.info("Agent stopped")
        
    def _user_ranking_loop(self):
        """Update user rankings periodically"""
        while self.running:
            try:
                logger.info("Starting user ranking update")
                start_time = time.time()
                
                # Build user graph
                self.graph_analyzer.build_graph(
                    seed_users=SEED_USERS,
                    max_users=MAX_USERS,
                    depth=2
                )
                
                # Get ranked users
                self.ranked_users = self.graph_analyzer.get_top_users(limit=MAX_USERS)
                
                # Calculate time to sleep (at least 24 hours, user rankings change slowly)
                elapsed = time.time() - start_time
                sleep_time = max(24 * 60 * 60, COLLECTION_INTERVAL * 10 - elapsed)
                
                logger.info(f"User ranking updated with {len(self.ranked_users)} users, sleeping for {sleep_time:.0f}s")
                
                # Sleep until next update
                time_slept = 0
                while self.running and time_slept < sleep_time:
                    time.sleep(min(60, sleep_time - time_slept))
                    time_slept += 60
                    
            except Exception as e:
                logger.error(f"Error in user ranking loop: {str(e)}")
                time.sleep(60)  # Sleep on error to avoid rapid retries
                
    def _user_collection_loop(self):
        """Main collection loop that runs periodically"""
        # Wait for user ranking to complete
        while self.running and not self.ranked_users:
            logger.info("Waiting for user ranking to complete...")
            time.sleep(10)
            
        # Start collection after user ranking
        while self.running:
            try:
                logger.info("Starting user-focused collection cycle")
                start_time = time.time()
                
                # Collect tweets from each target user
                processed_users = 0
                user_count = min(50, len(self.ranked_users))  # Process top 50 users per cycle
                
                for username, score in self.ranked_users[:user_count]:
                    self._collect_from_user(username, score)
                    processed_users += 1
                    
                    # Respect rate limits
                    time.sleep(2)
                    
                # Calculate time to sleep
                elapsed = time.time() - start_time
                sleep_time = max(1, COLLECTION_INTERVAL - elapsed)
                
                logger.info(f"Collection cycle completed: {processed_users} users processed in {elapsed:.2f}s, sleeping for {sleep_time:.2f}s")
                
                # Sleep until next collection cycle
                time_slept = 0
                while self.running and time_slept < sleep_time:
                    time.sleep(min(60, sleep_time - time_slept))
                    time_slept += 60
                    
            except Exception as e:
                logger.error(f"Error in user collection loop: {str(e)}")
                time.sleep(60)  # Sleep on error to avoid rapid retries
                
    def _keyword_search_loop(self):
        """Secondary collection using keyword searches"""
        # Wait for some data to be collected first
        time.sleep(5 * 60)  # Wait 5 minutes
        
        while self.running:
            try:
                logger.info("Starting keyword-based collection cycle")
                start_time = time.time()
                
                # Generate search queries
                queries = self.keyword_filter.generate_search_queries(
                    max_queries=5,
                    terms_per_query=3
                )
                
                if not queries:
                    logger.warning("No search queries generated, using default keywords")
                    # Use a few default combinations if no queries generated
                    default_terms = self.keyword_filter.lexicon[:10]  # Use top 10 terms
                    queries = [
                        f'"{default_terms[0]}" OR "{default_terms[1]}" OR "LLM security"',
                        f'"{default_terms[2]}" OR "{default_terms[3]}" OR "jailbreak"'
                    ]
                
                # Execute each search query
                total_results = 0
                for query in queries:
                    # Search for tweets
                    search_results = self.twitter_client.search_tweets(
                        query=query,
                        max_count=100
                    )
                    
                    logger.info(f"Search for '{query}' returned {len(search_results)} results")
                    total_results += len(search_results)
                    
                    # Process each result
                    for tweet in search_results:
                        # Check relevance using keyword filter
                        if hasattr(tweet, 'text'):
                            text = tweet.text
                        else:
                            text = tweet.get('text', '')
                            
                        relevance = self.keyword_filter.calculate_relevance(text)
                        
                        # If relevant, add to processing queue
                        if relevance >= KEYWORD_RELEVANCE_THRESHOLD:
                            self.queue.put({
                                "tweet": tweet,
                                "relevance": relevance,
                                "source": "keyword_search"
                            })
                            
                    # Respect rate limits
                    time.sleep(5)
                
                # Update statistics
                self.stats["tweets_collected"] += total_results
                
                # Calculate time to sleep
                elapsed = time.time() - start_time
                sleep_time = max(1, COLLECTION_INTERVAL - elapsed)
                
                logger.info(f"Keyword search completed: {total_results} tweets found in {elapsed:.2f}s, sleeping for {sleep_time:.2f}s")
                
                # Sleep until next search cycle
                time_slept = 0
                while self.running and time_slept < sleep_time:
                    time.sleep(min(60, sleep_time - time_slept))
                    time_slept += 60
                    
            except Exception as e:
                logger.error(f"Error in keyword search loop: {str(e)}")
                time.sleep(60)  # Sleep on error to avoid rapid retries
                
    def _collect_from_user(self, username, centrality_score):
        """
        Collect and process tweets from a user
        
        Args:
            username (str): Twitter username
            centrality_score (float): Centrality score from graph analysis
        """
        logger.info(f"Collecting from user {username} (score: {centrality_score:.3f})")
        
        try:
            # Get tweets from the user
            tweets = self.twitter_client.get_user_tweets(
                username=username,
                max_count=USER_TWEETS_LIMIT,
                include_replies=True
            )
            
            logger.info(f"Collected {len(tweets)} tweets from user {username}")
            
            # Update statistics
            self.stats["tweets_collected"] += len(tweets)
            
            # Process each tweet
            for tweet in tweets:
                # Store the tweet regardless of relevance (for historical data)
                self.storage.store_tweet(tweet)
                
                # Check relevance using keyword filter
                text = tweet.text if hasattr(tweet, 'text') else tweet.get('text', '')
                relevance = self.keyword_filter.calculate_relevance(text)
                
                # If relevant, add to processing queue
                if relevance >= KEYWORD_RELEVANCE_THRESHOLD:
                    self.queue.put({
                        "tweet": tweet,
                        "relevance": relevance,
                        "source": "user_timeline",
                        "centrality_score": centrality_score
                    })
                    
        except Exception as e:
            logger.error(f"Error collecting from user {username}: {str(e)}")
            
    def _processing_worker(self):
        """Worker thread for processing queue items"""
        while self.running:
            try:
                # Get item from queue with timeout to allow for shutdown
                item = self.queue.get(timeout=1)
                tweet = item["tweet"]
                source = item.get("source", "unknown")
                
                # Get tweet ID
                tweet_id = tweet.id if hasattr(tweet, 'id') else tweet.get('id')
                
                logger.info(f"Processing tweet {tweet_id} from {source}")
                
                try:
                    # Reconstruct conversation thread
                    conversation = self.twitter_client.reconstruct_conversation(
                        tweet,
                        max_depth=CONVERSATION_DEPTH
                    )
                    
                    # Store conversation
                    conversation_id = tweet.conversation_id if hasattr(tweet, 'conversation_id') else tweet_id
                    self.storage.store_conversation(conversation_id, conversation)
                    
                    # Analyze content using LLM
                    analysis_results = self.llm_analyzer.analyze_content(tweet, conversation)
                    
                    # Store analysis results
                    self.storage.store_analysis(tweet_id, analysis_results)
                    
                    # Update term statistics for keyword filtering
                    documents = []
                    for t in conversation.values():
                        if hasattr(t, 'text'):
                            documents.append(t.text)
                        else:
                            documents.append(t.get('text', ''))
                            
                    self.keyword_filter.update_statistics(documents)
                    
                    # Check if highly relevant
                    combined_score = analysis_results["scores"]["combined"]
                    
                    # Update statistics
                    self.stats["tweets_analyzed"] += 1
                    
                    if combined_score >= LLM_RELEVANCE_THRESHOLD:
                        logger.info(f"High relevance content detected (score: {combined_score:.2f}) from source: {source}")
                        self.stats["relevant_found"] += 1
                        
                        # Update lexicon based on high-relevance content
                        if self.stats["relevant_found"] % 10 == 0:  # Every 10 relevant finds
                            # Generate potential new terms
                            new_terms = self.keyword_filter.generate_new_terms()
                            
                            # Update lexicon
                            added_terms = self.keyword_filter.update_lexicon(new_terms)
                            
                            if added_terms:
                                logger.info(f"Added {len(added_terms)} new terms to lexicon")
                                self.stats["keywords_added"] += len(added_terms)
                                
                                # Store updated lexicon
                                self.storage.store_lexicon(self.keyword_filter.lexicon)
                    
                except Exception as e:
                    logger.error(f"Error processing tweet {tweet_id}: {str(e)}")
                
                # Mark task as done
                self.queue.task_done()
                
            except Exception as e:
                if str(e) != "Empty":  # Ignore timeout exceptions
                    logger.error(f"Error in processing worker: {str(e)}")
                
    def process_single_tweet(self, tweet_id):
        """
        Process a single tweet by ID (for testing or manual processing)
        
        Args:
            tweet_id (str): Twitter tweet ID
            
        Returns:
            dict: Analysis results
        """
        try:
            # Get the tweet
            tweet_response = self.twitter_client.client.get_tweet(
                tweet_id,
                tweet_fields=['id', 'text', 'author_id', 'created_at', 
                              'conversation_id', 'referenced_tweets']
            )
            
            if not tweet_response.data:
                logger.error(f"Tweet {tweet_id} not found")
                return None
                
            tweet = tweet_response.data
            
            # Store the tweet
            self.storage.store_tweet(tweet)
            
            # Reconstruct conversation thread
            conversation = self.twitter_client.reconstruct_conversation(
                tweet,
                max_depth=CONVERSATION_DEPTH
            )
            
            # Store conversation
            conversation_id = tweet.conversation_id if hasattr(tweet, 'conversation_id') else tweet_id
            self.storage.store_conversation(conversation_id, conversation)
            
            # Analyze content using LLM
            analysis_results = self.llm_analyzer.analyze_content(tweet, conversation)
            
            # Store analysis results
            self.storage.store_analysis(tweet_id, analysis_results)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error processing tweet {tweet_id}: {str(e)}")
            return None
```

### Command Line Interface

```python
# cli.py
import argparse
import json
import logging
import time
from tabulate import tabulate

from twitter_agent import TwitterAgent
from config import TWITTER_BEARER_TOKEN, LLM_PROVIDER, LLM_MODEL
from json_storage import JSONStorage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def start_agent():
    """Start the Twitter agent and keep it running"""
    agent = TwitterAgent()
    
    try:
        logger.info("Starting Twitter agent...")
        agent.start()
        
        # Keep the main thread alive
        while True:
            time.sleep(60)
            
            # Log statistics every hour
            if time.time() % 3600 < 60:  # Approximately every hour
                stats = agent.stats
                logger.info(f"Stats - Collected: {stats['tweets_collected']}, "
                           f"Analyzed: {stats['tweets_analyzed']}, "
                           f"Relevant: {stats['relevant_found']}, "
                           f"Keywords added: {stats['keywords_added']}")
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, stopping agent...")
        agent.stop()
        
def process_tweet(tweet_id):
    """Process a single tweet for testing"""
    agent = TwitterAgent()
    
    logger.info(f"Processing tweet {tweet_id}...")
    results = agent.process_single_tweet(tweet_id)
    
    if results:
        print("\nResults:")
        print("--------")
        print(f"Technical score: {results['scores']['technical']:.2f}")
        print(f"Security score: {results['scores']['security']:.2f}")
        print(f"LLM-specific score: {results['scores']['llm_specific']:.2f}")
        print(f"Combined score: {results['scores']['combined']:.2f}")
        print("\nInsights:")
        print(json.dumps(results["insights"], indent=2))
    else:
        print("Failed to process tweet.")
        
def list_tweets(min_score=0.6, limit=10):
    """List relevant tweets from the JSON storage"""
    storage = JSONStorage()
    tweets = storage.get_relevant_tweets(min_score, limit)
    
    if not tweets:
        print("No relevant tweets found.")
        return
        
    table_data = []
    for entry in tweets:
        tweet = entry.get("tweet", {})
        analysis = entry.get("analysis", {})
        
        text = tweet.get("text", "")
        if len(text) > 50:
            text = text[:47] + "..."
            
        table_data.append([
            tweet.get("id", ""),
            entry.get("relevance_score", 0.0),
            text,
            analysis.get("insights", {}).get("vulnerability_type", ""),
            time.strftime("%Y-%m-%d %H:%M", time.localtime(tweet.get("created_at", 0)))
        ])
        
    print(tabulate(
        table_data,
        headers=["Tweet ID", "Score", "Content", "Vulnerability Type", "Created"],
        tablefmt="fancy_grid"
    ))
    
def export_results(output_file="results.json", min_score=0.6, limit=50):
    """Export analysis results to a single JSON file"""
    storage = JSONStorage()
    tweets = storage.get_relevant_tweets(min_score, limit)
    
    if not tweets:
        print("No relevant tweets found to export.")
        return
        
    # Format results for export
    export_data = {
        "metadata": {
            "exported_at": time.time(),
            "min_score": min_score,
            "tweets_count": len(tweets),
            "llm_provider": LLM_PROVIDER,
            "llm_model": LLM_MODEL
        },
        "tweets": tweets
    }
    
    # Write to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)
        
    print(f"Exported {len(tweets)} tweets to {output_file}")
    
def show_lexicon():
    """Show current keyword lexicon"""
    storage = JSONStorage()
    lexicon = storage.get_lexicon()
    
    print("\nCurrent Keyword Lexicon")
    print("-----------------------")
    
    # Group terms by length for better readability
    short_terms = [term for term in lexicon if len(term) < 15]
    long_terms = [term for term in lexicon if len(term) >= 15]
    
    # Print short terms in columns
    rows = []
    for i in range(0, len(short_terms), 3):
        row = short_terms[i:i+3]
        while len(row) < 3:
            row.append("")
        rows.append(row)
            
    print(tabulate(rows, tablefmt="simple"))
    
    # Print long terms on separate lines
    if long_terms:
        print("\nComplex terms:")
        for term in long_terms:
            print(f"- {term}")
    
    print(f"\nTotal: {len(lexicon)} terms")
    
def show_config():
    """Show current configuration"""
    print("\nTwitter Agent Configuration")
    print("---------------------------")
    print(f"Twitter API: {'Configured' if TWITTER_BEARER_TOKEN else 'Not Configured'}")
    print(f"LLM Provider: {LLM_PROVIDER}")
    print(f"LLM Model: {LLM_MODEL}")
    print("\nUse --help for available commands")
    
def main():
    """Main entry point for CLI"""
    parser = argparse.ArgumentParser(description="Twitter Agent CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start the Twitter agent")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process a single tweet")
    process_parser.add_argument("tweet_id", help="Twitter tweet ID to process")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List relevant tweets")
    list_parser.add_argument(
        "--min-score", type=float, default=0.6,
        help="Minimum relevance score (default: 0.6)"
    )
    list_parser.add_argument(
        "--limit", type=int, default=10,
        help="Maximum number of tweets to show (default: 10)"
    )
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export results to JSON file")
    export_parser.add_argument(
        "--output", type=str, default="results.json",
        help="Output file path (default: results.json)"
    )
    export_parser.add_argument(
        "--min-score", type=float, default=0.6,
        help="Minimum relevance score (default: 0.6)"
    )
    export_parser.add_argument(
        "--limit", type=int, default=50,
        help="Maximum number of tweets to export (default: 50)"
    )
    
    # Lexicon command
    lexicon_parser = subparsers.add_parser("lexicon", help="Show current keyword lexicon")
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Show current configuration")
    
    args = parser.parse_args()
    
    if args.command == "start":
        start_agent()
    elif args.command == "process":
        process_tweet(args.tweet_id)
    elif args.command == "list":
        list_tweets(args.min_score, args.limit)
    elif args.command == "export":
        export_results(args.output, args.min_score, args.limit)
    elif args.command == "lexicon":
        show_lexicon()
    elif args.command == "config":
        show_config()
    else:
        show_config()
        
if __name__ == "__main__":
    main()
```

## Deployment Considerations

### Environment Setup

1. **Virtual Environment**: Use a dedicated virtual environment to isolate dependencies.

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

2. **Environment Variables**: Create a `.env` file with your API keys and configuration:

```
# Twitter API credentials
TWITTER_API_KEY=your_api_key
TWITTER_API_SECRET=your_api_secret
TWITTER_ACCESS_TOKEN=your_access_token
TWITTER_ACCESS_SECRET=your_access_secret
TWITTER_BEARER_TOKEN=your_bearer_token

# LLM API configuration
LLM_API_KEY=your_api_key
LLM_MODEL=gpt-4-turbo  # or claude-3-opus-20240229
LLM_PROVIDER=openai  # or anthropic

# Agent configuration
COLLECTION_INTERVAL=3600  # 1 hour
```

3. **Twitter API Access**: You'll need Twitter API access (ideally Academic Research or Elevated access) for the best results, particularly for comprehensive conversation reconstruction.

### Rate Limiting Considerations

1. **Twitter API Rate Limits**:
   - Standard API: 500,000 tweets/month, various endpoint-specific limits
   - Elevated access: Higher limits, more endpoints
   - Academic Research: Highest limits, full archive search

2. **Mitigation Strategies**:
   - Implement adaptive throttling
   - Use exponential backoff for retries
   - Distribute requests across time windows
   - Prioritize high-value users

3. **LLM API Costs**:
   - Monitor token usage
   - Implement caching for similar analyses
   - Consider pre-filtering with smaller models

## Examples and Usage

### Basic Usage

Starting the agent:

```bash
python cli.py start
```

Processing a single tweet:

```bash
python cli.py process 1234567890
```

Listing relevant tweets:

```bash
python cli.py list --min-score 0.7 --limit 20
```

Viewing the current lexicon:

```bash
python cli.py lexicon
```

Exporting results to a JSON file:

```bash
python cli.py export --output results.json --min-score 0.7 --limit 100
```

### Example Analysis Output

Here's an example of the analysis output for a tweet discussing an LLM vulnerability:

```json
{
  "scores": {
    "technical": 0.85,
    "security": 0.92,
    "llm_specific": 0.88,
    "combined": 0.88
  },
  "insights": {
    "vulnerability_type": "Prompt injection via Unicode manipulation",
    "target_models": ["GPT-4", "Claude-2"],
    "effectiveness": "High for GPT-4, moderate for Claude-2",
    "novelty": "Novel technique combining existing methods",
    "key_techniques": [
      "Unicode homoglyph substitution",
      "Control character insertion",
      "Multi-stage payload delivery"
    ],
    "potential_mitigations": [
      "Unicode normalization",
      "Control character filtering",
      "Input pattern detection"
    ]
  },
  "timestamp": 1686744568
}
```

### Directory Structure

The recommended project directory structure:

```
twitter-agent/
│
├── data/                      # JSON data storage
│   ├── tweets/                # Individual tweet JSON files
│   ├── conversations/         # Conversation threads
│   ├── analysis/              # Analysis results
│   ├── lexicon/               # Lexicon versions
│   └── index.json             # Index of all tweets with metadata
│
├── logs/                      # Log files
│   └── twitter_agent.log      # Main log file
│
├── config.py                  # Configuration settings
├── graph_analyzer.py          # Graph centrality analysis
├── twitter_client.py          # Twitter API client
├── keyword_filter.py          # Keyword-based filtering
├── llm_analyzer.py            # LLM-based content analysis
├── json_storage.py            # JSON file storage
├── twitter_agent.py           # Main agent implementation
├── cli.py                     # Command-line interface
│
├── requirements.txt           # Project dependencies
├── .env                       # Environment variables (not in version control)
├── .gitignore                 # Git ignore file
└── README.md                  # Project documentation
```

## Further Enhancements

The Twitter/X Agent can be extended in several ways:

1. **Integration with notification systems**: Send alerts for high-relevance findings via email, Slack, or other channels.

2. **Web dashboard**: Create a web interface for monitoring findings and trends.

3. **Enhanced graph analysis**: Incorporate more advanced centrality measures or community detection algorithms.

4. **Cross-platform integration**: Combine with agents monitoring other platforms (like the Reddit Agent) for comprehensive coverage.

5. **Temporal analysis**: Add visualization tools to track vulnerability evolution over time.

6. **Improved conversation reconstruction**: Enhance the conversation thread building with more sophisticated algorithms or by using better API access.

7. **Multi-LLM consensus**: Use multiple LLMs in parallel to achieve better consensus on vulnerability assessment.

## Troubleshooting

### Common Issues

1. **Twitter API Rate Limiting**:
   - *Symptom*: 429 Too Many Requests errors
   - *Solution*: Implement exponential backoff, reduce collection frequency, or apply for higher API access level

2. **LLM API Errors**:
   - *Symptom*: Timeout or quota exceeded errors
   - *Solution*: Implement retries with backoff, check API key validity, or adjust request volumes

3. **Graph Building Failures**:
   - *Symptom*: Empty or small user graphs
   - *Solution*: Check seed users validity, ensure API has proper permissions, increase traversal depth

4. **High False Positive Rate**:
   - *Symptom*: Too many irrelevant tweets flagged as relevant
   - *Solution*: Adjust keyword and LLM relevance thresholds, refine LLM prompts, or expand lexicon

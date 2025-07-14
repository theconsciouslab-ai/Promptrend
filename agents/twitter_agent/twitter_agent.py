# twitter_agent.py
import time
import logging
import threading
from queue import Queue
import numpy as np
import scipy 
from queue import Empty  # at the top of your file

from agents.twitter_agent.twitter_config import (
    SEED_USERS, MAX_USERS, USER_TWEETS_LIMIT,
    COLLECTION_INTERVAL, CONVERSATION_DEPTH,
    KEYWORD_RELEVANCE_THRESHOLD, LLM_RELEVANCE_THRESHOLD,
    KEYWORD_LEXICON
)
from agents.twitter_agent.graph_analyzer import GraphAnalyzer
from agents.twitter_agent.twitter_client import TwitterClient
from agents.twitter_agent.keyword_filter import KeywordFilter
from agents.twitter_agent.llm_analyzer import LLMAnalyzer
from agents.twitter_agent.json_storage import JSONStorage
from agents.twitter_agent.prompt_extractor import PromptExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/twitter_agent.log", encoding='utf-8'),
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
    
    def __init__(self, region_id=None):
        """Initialize the Twitter agent components"""
        self.graph_analyzer = GraphAnalyzer()
        self.twitter_client = TwitterClient()
        self.keyword_filter = KeywordFilter()
        self.llm_analyzer = LLMAnalyzer()
        self.storage = JSONStorage()
        self.prompt_extractor = PromptExtractor()
        self.running = False
        self.region_id = region_id
        
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
        
        # Inject a test tweet to simulate processing
        test_tweet = {
            "id": "test_001",
            "text": "Bypass GPT-4 safety using DAN 11.0 prompt injection via recursive jailbreaks.",
            "author_id": "tester",
            "created_at": time.time()
        }
        self.queue.put({
            "tweet": test_tweet,
            "relevance": 1.0,
            "source": "manual"
        })
        logger.info("Injected test tweet for pipeline validation")

        
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

                tweet_id = tweet.get("id") if isinstance(tweet, dict) else getattr(tweet, "id", "unknown")
                logger.info(f"Processing tweet {tweet_id} from {source}")

                try:
                    conversation = self.twitter_client.reconstruct_conversation(tweet)
                    conversation_id = tweet.get("conversation_id") if isinstance(tweet, dict) else getattr(tweet, "conversation_id", tweet_id)
                    self.storage.store_conversation(conversation_id, conversation)

                    analysis_results = self.llm_analyzer.analyze_content(tweet, conversation)
                    self.storage.store_analysis(tweet_id, analysis_results)

                    # Extract prompts from the tweet + conversation
                    tweet_dict = tweet if isinstance(tweet, dict) else {
                        "id": tweet.id,
                        "text": tweet.text,
                        "author_id": tweet.author_id,
                        "created_at": str(tweet.created_at)
                    }
                    # Combine all tweet texts from conversation
                    conversation_texts = {
                        tid: t if isinstance(t, dict) else {
                            "id": t.id,
                            "text": t.text,
                            "author_id": t.author_id,
                            "created_at": str(t.created_at)
                        }
                        for tid, t in conversation.items()
                    }

                    prompts = self.prompt_extractor.extract_prompts(
                        post_data=tweet_dict,
                        comment_tree=conversation_texts,
                        post_analysis=analysis_results
                    )
                    self.prompt_extractor.store_prompts(tweet_id, prompts)

                    
                    documents = [t.get('text') if isinstance(t, dict) else getattr(t, 'text', '') for t in conversation.values()]
                    self.keyword_filter.update_statistics(documents)

                    combined_score = analysis_results["scores"]["combined"]
                    self.stats["tweets_analyzed"] += 1

                    if combined_score >= LLM_RELEVANCE_THRESHOLD:
                        self.stats["relevant_found"] += 1
                        logger.info(f"Relevant tweet {tweet_id} detected (score: {combined_score:.2f})")

                except Exception as e:
                    logger.error(f"Error processing tweet {tweet_id}: {str(e)}", exc_info=True)

                self.queue.task_done()

            except Empty:
                continue  # Normal behavior, skip

            except Exception as e:
                logger.error(f"Unexpected error in worker loop: {str(e)}", exc_info=True)

                
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
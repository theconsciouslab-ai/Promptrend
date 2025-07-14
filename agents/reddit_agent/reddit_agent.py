# reddit_agent.py
import time
import logging
import threading
import os
import traceback
import glob
import json
import sys


from queue import Queue , Empty


from agents.reddit_agent.reddit_config import (
    TARGET_SUBREDDITS, COLLECTION_INTERVAL,
    KEYWORD_RELEVANCE_THRESHOLD, LLM_RELEVANCE_THRESHOLD,
    COMMENT_DEPTH
)
from agents.reddit_agent.reddit_client import RedditClient
from agents.reddit_agent.keyword_filter  import KeywordFilter
from agents.reddit_agent.llm_analyzer import LLMAnalyzer
from agents.reddit_agent.json_storage import JSONStorage




logger = logging.getLogger(__name__)

class RedditAgent:
    """
    Agent for monitoring Reddit discussions related to LLM security
    
    This agent collects posts from targeted subreddits, filters them using a
    two-stage process (keyword filtering followed by LLM analysis), and stores
    relevant content as JSON files for further analysis.
    """
    
    def __init__(self, region_id=None):
        """Initialize the Reddit agent components"""
        self.reddit_client = RedditClient()
        self.keyword_filter = KeywordFilter()
        self.llm_analyzer = LLMAnalyzer()
        self.region_id = region_id
        self.storage = JSONStorage()
        self.running = False
        
        
        # Processing queue for handling posts asynchronously
        self.queue = Queue()
        logger.info("✅ RedditAgent initialized — logging is active.")

        
        logger.info("Reddit agent initialized")
        
        

        
    def start(self):
        """Start the agent's collection and processing threads"""
        if self.running:
            logger.warning("Agent is already running")
            return
            
        self.running = True
        
        # Start collection thread
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            daemon=True
        )
        self.collection_thread.start()
        
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
        if hasattr(self, 'collection_thread'):
            self.collection_thread.join(timeout=5)
            
        for thread in getattr(self, 'worker_threads', []):
            thread.join(timeout=5)
            
        logger.info("Agent stopped")
        
    def _collection_loop(self):
        """Main collection loop that runs periodically"""
        while self.running:
            try:
                logger.info("Starting collection cycle")
                start_time = time.time()
                
                # Collect posts from each target subreddit
                for subreddit in TARGET_SUBREDDITS:
                    self._collect_from_subreddit(subreddit)
                    
                # Calculate time to sleep
                elapsed = time.time() - start_time
                sleep_time = max(1, COLLECTION_INTERVAL - elapsed)
                
                # Retry previously skipped posts using updated lexicon
                logger.info("🔄 Automatically retrying skipped posts after this cycle...")
                self.retry_skipped()
                
                logger.info(f"Collection cycle completed in {elapsed:.2f}s, sleeping for {sleep_time:.2f}s")
                
                # Sleep until next collection cycle
                logger.info(f"⏱ Waiting for next collection cycle — sleeping {sleep_time:.2f} seconds.")
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in collection loop: {str(e)}")
                time.sleep(60)  # Sleep on error to avoid rapid retries
                
    def _collect_from_subreddit(self, subreddit):
        """
        Collect and process posts from a subreddit
        
        Args:
            subreddit (str): Name of the subreddit to collect from
        """
        logger.info(f"Collecting from r/{subreddit}")
        
        try:
            # Get submissions from the subreddit
            submissions = self.reddit_client.get_submissions(subreddit)
            logger.info(f"Collected {len(submissions)} submissions from r/{subreddit}")
            
            # Process each submission
            for submission in submissions:
                # Store the post regardless of relevance (for historical data)
                self.storage.store_post(submission)
                
                # Check relevance using keyword filter
                score_title = self.keyword_filter.calculate_relevance(submission.title)
                score_body = self.keyword_filter.calculate_relevance(submission.selftext)
                # Calculate relevance
                relevance = 0.6 * score_title + 0.4 * score_body

                # Force-pass heuristic for obvious prompt content
                rescue_keywords = ["prompt", "chatgpt", "write", "unlock", "jailbreak", "nsfw", "story", "answer"]
                is_prompt_rich = (
                    "prompt" in submission.selftext.lower() and 
                    any(kw in submission.selftext.lower() for kw in rescue_keywords)
                )
                
                logger.debug(f"Submission {submission.id} keyword relevance: {relevance:.2f}")
                logger.info(f"Submission {submission.id} relevance: {relevance:.2f} — Title: {submission.title}")
                
                # If relevant, add to processing queue
                if relevance >= KEYWORD_RELEVANCE_THRESHOLD or is_prompt_rich:
                    self.queue.put({
                        "submission": submission,
                        "relevance": relevance
                    })
                elif 0.05 <= relevance < KEYWORD_RELEVANCE_THRESHOLD and "prompt" in submission.selftext.lower():
                    logger.info(f"📥 Bypassing threshold for potential prompt-rich post: {submission.permalink}")
                    self.queue.put({
                        "submission": submission,
                        "relevance": relevance
                    })
                else:
                    # 👇 Store for later retry
                    self.storage.store_skipped_post(submission)
                     
        except Exception as e:
            logger.error(f"Error collecting from r/{subreddit}: {str(e)}")
            
    def _processing_worker(self):
        """Worker thread for processing queue items"""
        while self.running:
            try:
                # Get item from queue with timeout to allow for shutdown
                item = self.queue.get(timeout=1)
                submission = item["submission"]
                
                logger.info(f"Processing submission {submission.id} from r/{submission.subreddit.display_name}")
                
                # Get comment tree
                comment_tree = self.reddit_client.get_comments(submission, COMMENT_DEPTH)
                
                # Store comments
                self.storage.store_comments(submission.id, comment_tree)
                
                # Analyze content using LLM
                analysis_results = self.llm_analyzer.analyze_content(
                    {
                        "id": submission.id,
                        "title": submission.title,
                        "selftext": submission.selftext
                    },
                    comment_tree
                )
                
                # Store analysis results
                self.storage.store_analysis(submission.id, analysis_results)
                # Log analysis results
                logger.info(f"Analysis results for submission {submission.id}: {analysis_results}")
                
                
                # Check if highly relevant
                combined_score = analysis_results["scores"]["combined"]
                
                
                if combined_score >= LLM_RELEVANCE_THRESHOLD:
                    logger.info(f"High relevance content detected (score: {combined_score:.2f}): {submission.permalink}")
                    vulnerability_record = {
                        "id": submission.id,
                        "subreddit": getattr(submission.subreddit, "display_name", "unknown"),
                        "title": submission.title,
                        "selftext": submission.selftext,
                        "author": submission.author.name if submission.author else "[deleted]",
                        "created_utc": getattr(submission, "created_utc", int(time.time())),
                        "score": getattr(submission, "score", 0),
                        "url": getattr(submission, "url", ""),
                        "permalink": getattr(submission, "permalink", ""),
                        "num_comments": getattr(submission, "num_comments", 0),
                        "collected_at": int(time.time()),
                        "analysis": analysis_results,
                        "relevance_score": combined_score,
                        "is_vulnerability": True,
                        "platform": "reddit"
                    }
                    self.storage.store_vulnerability(vulnerability_record)
                    logger.info(f"🔍 Expanding lexicon from post {submission.id}")  
                    # Mark task as done
                    # Get prompts to feed into LLM
                    prompt_examples = [submission.selftext]

                    # Add up to 3 high-score comments
                    top_comments = sorted(comment_tree.values(), key=lambda x: x.get("score", 0), reverse=True)[:3]
                    prompt_examples += [c["body"] for c in top_comments if len(c.get("body", "")) > 20]

                    # Ask the LLM for new keywords
                    self.keyword_filter.expand_lexicon(prompt_examples)

                    # Apply them to the active lexicon
                    new_terms = list(self.keyword_filter.emergent_terms)
                    added_terms = self.keyword_filter.update_lexicon(new_terms)
                    self.keyword_filter.save_lexicon()
                    logger.info(f"➕ Added {len(added_terms)} new keyword(s) to the lexicon")
                self.queue.task_done()
                
            except Empty:
                logger.info("No items in queue — skipping.")
            except Exception as e:
                logger.error(f"Error processing submission: {str(e)}")
                logger.error(traceback.format_exc())

    def process_single_post(self, post_id):
        """
        Process a single post by ID (for testing or manual processing)
        
        Args:
            post_id (str): Reddit post ID
            
        Returns:
            dict: Analysis results
        """
        try:
            # Get the submission
            submission = self.reddit_client.reddit.submission(id=post_id)
            
            # Store the post
            self.storage.store_post(submission)
            
            # Get comment tree
            comment_tree = self.reddit_client.get_comments(submission, COMMENT_DEPTH)
            
            # Store comments
            self.storage.store_comments(submission.id, comment_tree)
            
            # Analyze content using LLM
            analysis_results = self.llm_analyzer.analyze_content(
                {
                    "id": submission.id,
                    "title": submission.title,
                    "selftext": submission.selftext
                },
                comment_tree
            )
            
            # Store analysis results
            self.storage.store_analysis(submission.id, analysis_results)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error processing post {post_id}: {str(e)}")
            return None
        
        

    def retry_skipped(self):
        """
        Re-evaluate previously skipped posts using the updated keyword filter.
        If a post now meets the relevance threshold, queue it for full analysis.
        Posts that have been retried 5 times will be automatically deleted for performance.
        """
        skipped_dir = os.path.join(self.storage.data_dir, "skipped")
        if not os.path.exists(skipped_dir):
            logger.info("No skipped posts to retry.")
            return

        skipped_files = glob.glob(os.path.join(skipped_dir, "*.json"))
        if not skipped_files:
            logger.info("Skipped folder is empty.")
            return

        rescued = 0
        deleted = 0
        total_skipped = len(skipped_files)
        logger.info(f"🔄 Retrying {total_skipped} skipped posts...")

        for file_path in skipped_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    post_data = json.load(f)

                post_id = post_data.get("id")
                
                # Check if we should delete this post (5+ retries)
                if not self.storage.increment_retry_count(post_id):
                    deleted += 1
                    continue  # Post was deleted due to retry limit

                text_content = f"{post_data.get('title', '')} {post_data.get('selftext', '')}"
                relevance = self.keyword_filter.calculate_relevance(text_content)

                if relevance >= KEYWORD_RELEVANCE_THRESHOLD:
                    from types import SimpleNamespace
                    fake_submission = SimpleNamespace(
                        id=post_data["id"],
                        title=post_data.get("title", ""),
                        selftext=post_data.get("selftext", ""),
                        created_utc=post_data.get("created_utc", int(time.time())),
                        subreddit=SimpleNamespace(display_name=post_data.get("subreddit", "unknown")),
                        permalink=post_data.get("permalink", ""),
                        score=post_data.get("score", 0),
                        num_comments=post_data.get("num_comments", 0)
                    )

                    logger.info(f"🔄 Resurrecting skipped post {fake_submission.id} (score: {relevance:.2f})")
                    real_submission = self.reddit_client.reddit.submission(id=post_data["id"])
                    self.queue.put({"submission": real_submission, "relevance": relevance})
                    rescued += 1

                    # Remove from skipped folder since we're processing it
                    os.remove(file_path)

            except Exception as e:
                logger.error(f"Failed to retry skipped post from {file_path}: {str(e)}")

        # ✅ Log summary only once at the end
        logger.info(f"🔄 Retry summary: rescued {rescued}, deleted {deleted} (5+ retries) out of {total_skipped} skipped posts this cycle.")
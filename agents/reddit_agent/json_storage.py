# json_storage.py
import json
import os
import time
import logging
from pathlib import Path
from agents.reddit_agent.reddit_config import DATA_DIR

logger = logging.getLogger(__name__)

class JSONStorage:
    """Storage manager for saving Reddit posts and analysis results as JSON files"""
    
    def __init__(self, data_dir=None):
        """
        Initialize the JSON storage
        
        Args:
            data_dir (str, optional): Directory to store JSON files
        """
        self.data_dir =  DATA_DIR
        
        # Create data directories with the new structure
        self.posts_dir = os.path.join(self.data_dir, "posts")
        self.vulnerabilities_dir = os.path.join(self.data_dir, "vulnerabilities")
        self.comments_dir = os.path.join(self.data_dir, "comments")
        self.analysis_dir = os.path.join(self.data_dir, "analysis")
        self.index_file = os.path.join(self.data_dir, "index.json")
        self.prompts_dir = os.path.join(self.data_dir, "prompts")
        self.skipped_dir = os.path.join(self.data_dir, "skipped")

        
        Path(self.posts_dir).mkdir(parents=True, exist_ok=True)
        Path(self.comments_dir).mkdir(parents=True, exist_ok=True)
        Path(self.analysis_dir).mkdir(parents=True, exist_ok=True)
        Path(self.vulnerabilities_dir).mkdir(parents=True, exist_ok=True)
        Path(self.prompts_dir).mkdir(parents=True, exist_ok=True)
        Path(self.skipped_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize index if it doesn't exist
        if not os.path.exists(self.index_file):
            self._write_json(self.index_file, {
                "posts": {},
                "last_updated": int(time.time())
            })
            
        logger.info(f"JSON storage initialized at {self.data_dir}")
        
    def store_post(self, post):
        """
        Store a Reddit post as a JSON file
        
        Args:
            post: Reddit submission object
            
        Returns:
            bool: Success status
        """
        try:
            # Create a clean dictionary manually instead of converting the PRAW object directly
            post_data = {}
            
            # Safely extract attributes one by one
            try: post_data["id"] = post.id
            except: post_data["id"] = "unknown"
            
            try: post_data["subreddit"] = post.subreddit.display_name
            except: post_data["subreddit"] = "unknown"
            
            try: post_data["title"] = post.title
            except: post_data["title"] = ""
            
            try: 
                # Handle potential UTF-8 encoding issues in selftext
                selftext = post.selftext
                # Clean or truncate problematic text if needed
                if len(selftext) > 100000:  # If extremely long, truncate
                    selftext = selftext[:100000] + "... [truncated]"
                post_data["selftext"] = selftext
            except:
                post_data["selftext"] = ""
            
            try: post_data["author"] = post.author.name if post.author else "[deleted]"
            except: post_data["author"] = "[unknown]"
            
            try: post_data["created_utc"] = post.created_utc
            except: post_data["created_utc"] = int(time.time())
            
            try: post_data["score"] = post.score
            except: post_data["score"] = 0
            
            try: post_data["url"] = post.url
            except: post_data["url"] = ""
            
            try: post_data["permalink"] = post.permalink
            except: post_data["permalink"] = ""
            
            try: post_data["num_comments"] = post.num_comments
            except: post_data["num_comments"] = 0
            
            post_data["collected_at"] = int(time.time())
            
            # Save post data using the _write_json utility method
            post_file = os.path.join(self.posts_dir, f"{post.id}.json")
            self._write_json(post_file, post_data)
            
            # Update index with simplified data
            try:
                self._update_index(post_data)
            except Exception as e:
                logger.warning(f"Error updating index for post {post.id}: {str(e)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing post {post.id if hasattr(post, 'id') else 'unknown'}: {str(e)}")
            return False

    def store_comments(self, post_id, comment_tree):
        """
        Store comment tree as a JSON file
        
        Args:
            post_id (str): Post ID
            comment_tree (dict): Comment tree data
            
        Returns:
            bool: Success status
        """
        try:
            # Create a clean dictionary for comments
            clean_comments = {}
            
            # Process each comment individually
            for comment_id, comment_data in comment_tree.items():
                clean_comment = {}
                
                # Only include essential fields
                for key in ["id", "body", "author", "score", "created_utc"]:
                    if key in comment_data:
                        # For 'body', clean potential problematic content
                        if key == "body":
                            try:
                                body = comment_data[key]
                                # Truncate very long comments
                                if len(body) > 10000:
                                    body = body[:10000] + "... [truncated]"
                                clean_comment[key] = body
                            except:
                                clean_comment[key] = "[content error]"
                        else:
                            clean_comment[key] = comment_data[key]
                
                clean_comments[comment_id] = clean_comment
            
            # Save comment tree using the _write_json utility method
            comments_file = os.path.join(self.comments_dir, f"{post_id}.json")
            self._write_json(comments_file, {
                "post_id": post_id,
                "comments": clean_comments,
                "collected_at": int(time.time())
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing comments for post {post_id}: {str(e)}")
            return False

    def store_analysis(self, post_id, analysis_results):
        """
        Store analysis results as a JSON file
        
        Args:
            post_id (str): Post ID
            analysis_results (dict): Analysis results
            
        Returns:
            bool: Success status
        """
        try:
            # Create a clean dictionary for analysis
            clean_analysis = {"post_id": post_id}
            
            # Safely extract scores
            if isinstance(analysis_results, dict) and "scores" in analysis_results:
                scores = analysis_results["scores"]
                clean_scores = {}
                
                # Extract each score value safely
                for score_type in ["technical", "security", "llm_specific", "combined"]:
                    if score_type in scores and isinstance(scores[score_type], (int, float)):
                        clean_scores[score_type] = float(scores[score_type])
                    else:
                        clean_scores[score_type] = 0.0
                        
                clean_analysis["scores"] = clean_scores
            else:
                clean_analysis["scores"] = {
                    "technical": 0.0,
                    "security": 0.0,
                    "llm_specific": 0.0,
                    "combined": 0.0
                }
            
            # Safely extract insights
            if isinstance(analysis_results, dict) and "insights" in analysis_results:
                insights = analysis_results["insights"]
                clean_insights = {}
                
                # Process string fields
                for field in ["vulnerability_type", "effectiveness", "novelty"]:
                    if field in insights and isinstance(insights[field], str):
                        clean_insights[field] = insights[field][:500]  # Limit length
                    else:
                        clean_insights[field] = None
                
                # Process list fields
                for field in ["target_models", "key_techniques", "potential_mitigations"]:
                    if field in insights and isinstance(insights[field], list):
                        clean_list = []
                        for item in insights[field]:
                            if isinstance(item, str):
                                clean_list.append(item[:200])  # Limit length of each item
                        clean_insights[field] = clean_list
                    else:
                        clean_insights[field] = []
                        
                clean_analysis["insights"] = clean_insights
            else:
                clean_analysis["insights"] = {
                    "vulnerability_type": None,
                    "target_models": [],
                    "effectiveness": None,
                    "novelty": None,
                    "key_techniques": [],
                    "potential_mitigations": []
                }
            
            # Add timestamp
            clean_analysis["timestamp"] = int(time.time())
            
            # Save analysis results using the _write_json utility method
            analysis_file = os.path.join(self.analysis_dir, f"{post_id}.json")
            self._write_json(analysis_file, clean_analysis)
            
            # Update index with score
            combined_score = clean_analysis["scores"]["combined"]
            self._update_index_score(post_id, combined_score)
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing analysis for post {post_id}: {str(e)}")
            return False
        
    def store_vulnerability(self, item):
        """
        Store a Reddit vulnerability as a JSON file in the vulnerabilities folder.

        Args:
            item (dict): The vulnerability record (must have 'id' or 'post_id')
        Returns:
            bool: Success status
        """
        try:
            post_id = str(item.get("id") or item.get("post_id") or int(time.time() * 1000))
            safe_post_id = post_id.replace("/", "_")
            file_path = os.path.join(self.vulnerabilities_dir, f"{safe_post_id}.json")
            self._write_json(file_path, item)
            return True
        except Exception as e:
            logger.error(f"Error storing vulnerability {item.get('id', 'unknown')}: {str(e)}")
            return False

        
    def store_skipped_post(self, post):
        """
        Store a skipped post with retry counter, delete after 5 retries
        """
        try:
            post_id = post.id
            skipped_file = os.path.join(self.skipped_dir, f"{post_id}.json")
            
            # Check if post was already skipped
            if os.path.exists(skipped_file):
                with open(skipped_file, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                retry_count = existing_data.get("retry_count", 0) + 1
                first_skipped = existing_data.get("first_skipped", int(time.time()))
            else:
                retry_count = 1
                first_skipped = int(time.time())
            
            # Delete after 5 retries or 7 days old
            should_delete = (
                retry_count >= 5 or  # 5+ retries - PERFORMANCE OPTIMIZATION
                (time.time() - first_skipped) > (7 * 24 * 3600) or  # 7+ days old
                self._is_obviously_irrelevant(post)  # Obviously not AI-related
            )
            
            if should_delete:
                # Delete the file to improve performance
                if os.path.exists(skipped_file):
                    os.remove(skipped_file)
                logger.debug(f"🗑️ Deleted post {post_id} after {retry_count} attempts (performance optimization)")
                return True
            
            # Store with minimal data (just what we need for retry)
            post_data = {
                "id": post.id,
                "title": post.title[:200],  # Truncated
                "selftext": post.selftext[:500],  # Truncated  
                "created_utc": getattr(post, "created_utc", int(time.time())),
                "subreddit": post.subreddit.display_name if post.subreddit else "unknown",
                "retry_count": retry_count,
                "first_skipped": first_skipped,
                "last_skipped": int(time.time())
            }
            
            with open(skipped_file, "w", encoding="utf-8") as f:
                json.dump(post_data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"📥 Skipped post {post_id} (attempt {retry_count}/5)")
            return True
            
        except Exception as e:
            logger.error(f"Error handling skipped post {post.id}: {str(e)}")
            return False
    
    def increment_retry_count(self, post_id):
        """
        Increment retry count for a skipped post and delete if >= 5 retries
        
        Args:
            post_id (str): Post ID to increment retry count for
            
        Returns:
            bool: True if post still exists, False if deleted
        """
        try:
            skipped_file = os.path.join(self.skipped_dir, f"{post_id}.json")
            
            if not os.path.exists(skipped_file):
                return False
            
            with open(skipped_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Increment retry count
            retry_count = data.get("retry_count", 0) + 1
            data["retry_count"] = retry_count
            data["last_retried"] = int(time.time())
            
            # Delete if reached 5 retries
            if retry_count >= 5:
                os.remove(skipped_file)
                logger.debug(f"🗑️ Deleted post {post_id} after {retry_count} retry attempts (performance optimization)")
                return False
            
            # Update the file with new retry count
            with open(skipped_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"📈 Incremented retry count for post {post_id} to {retry_count}/5")
            return True
            
        except Exception as e:
            logger.error(f"Error incrementing retry count for post {post_id}: {str(e)}")
            return False
    
    def _is_obviously_irrelevant(self, post):
        """Quick check for obviously irrelevant content"""
        combined_text = f"{post.title} {post.selftext}".lower()
        
        # If it has AI terms, keep it for now
        ai_terms = ["ai", "llm", "gpt", "chatgpt", "claude", "prompt", "model", "openai", "anthropic"]
        if any(term in combined_text for term in ai_terms):
            return False
        
        # If it's very short and no AI context, probably irrelevant
        if len(combined_text.split()) < 10:
            return True
        
        # Common irrelevant categories
        irrelevant_terms = [
            "buy", "sell", "price", "stock", "crypto", "bitcoin", "trading",
            "dating", "relationship", "girlfriend", "boyfriend",
            "game", "gaming", "minecraft", "fortnite", "xbox", "playstation",
            "food", "recipe", "cooking", "restaurant", "pizza",
            "movie", "netflix", "tv show", "actor", "celebrity",
            "sports", "football", "basketball", "soccer", "baseball"
        ]
        
        irrelevant_count = sum(1 for term in irrelevant_terms if term in combined_text)
        
        # If lots of irrelevant terms and no AI context, delete it
        return irrelevant_count >= 3

    def cleanup_old_skipped_files(self):
        """Performance cleanup: delete files with 5+ retries or older than 7 days"""
        try:
            cutoff_time = time.time() - (7 * 24 * 3600)  # 7 days ago
            deleted_count = 0
            
            for filename in os.listdir(self.skipped_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.skipped_dir, filename)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        first_skipped = data.get('first_skipped', time.time())
                        retry_count = data.get('retry_count', 0)
                        
                        # Delete if 5+ retries OR older than 7 days
                        if retry_count >= 5 or first_skipped < cutoff_time:
                            os.remove(file_path)
                            deleted_count += 1
                            
                    except Exception as e:
                        logger.warning(f"Error processing {file_path}: {str(e)}")
                        # If we can't read it, just delete it
                        os.remove(file_path)
                        deleted_count += 1
            
            if deleted_count > 0:
                logger.info(f"🗑️ Performance cleanup: deleted {deleted_count} old/over-retried skipped files")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

            
    def get_post_with_comments(self, post_id):
        """
        Get post and comments from storage
        
        Args:
            post_id (str): Post ID
            
        Returns:
            dict: Post data with comments and analysis
        """
        try:
            # Get post data
            post_file = os.path.join(self.posts_dir, f"{post_id}.json")
            if not os.path.exists(post_file):
                return None
                
            post_data = self._read_json(post_file)
            
            # Get comments
            comments_file = os.path.join(self.comments_dir, f"{post_id}.json")
            comments_data = self._read_json(comments_file) if os.path.exists(comments_file) else {"comments": {}}
            
            # Get analysis results
            analysis_file = os.path.join(self.analysis_dir, f"{post_id}.json")
            analysis_data = self._read_json(analysis_file) if os.path.exists(analysis_file) else None
            
            # Combine everything
            result = {
                "post": post_data,
                "comments": comments_data.get("comments", {}),
                "analysis": analysis_data
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving post {post_id}: {str(e)}")
            return None
            
    def get_relevant_posts(self, min_score=0.6, limit=50):
        """
        Get posts with high relevance scores
        
        Args:
            min_score (float): Minimum combined score
            limit (int): Maximum number of posts to return
            
        Returns:
            list: List of relevant posts
        """
        try:
            # Read index
            index_data = self._read_json(self.index_file)
            posts_index = index_data.get("posts", {})
            
            # Filter posts by score
            relevant_posts = []
            
            for post_id, post_info in posts_index.items():
                if post_info.get("score", 0.0) >= min_score:
                    # Read post data
                    post_file = os.path.join(self.posts_dir, f"{post_id}.json")
                    if os.path.exists(post_file):
                        post_data = self._read_json(post_file)
                        post_data["relevance_score"] = post_info.get("score", 0.0)
                        
                        # Add analysis insights if available
                        analysis_file = os.path.join(self.analysis_dir, f"{post_id}.json")
                        if os.path.exists(analysis_file):
                            analysis_data = self._read_json(analysis_file)
                            post_data["insights"] = analysis_data.get("insights", {})
                            


                            
                        relevant_posts.append(post_data)
            
            # Sort by score (descending) and collection time (descending)
            relevant_posts.sort(
                key=lambda x: (x.get("relevance_score", 0.0), x.get("collected_at", 0)), 
                reverse=True
            )
            
            # Limit results
            return relevant_posts[:limit]
            
        except Exception as e:
            logger.error(f"Error retrieving relevant posts: {str(e)}")
            return []
    

    def _write_json(self, file_path, data):
        """
        Write data to a JSON file
        
        Args:
            file_path (str): Path to JSON file
            data (dict): Data to write
        """
        # Sanitize the data to ensure it's JSON serializable
        sanitized_data = self._sanitize_json_data(data)
        
        try:
            # Write sanitized data to file with proper encoding and handling
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(sanitized_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error writing to {file_path}: {str(e)}")

    def _sanitize_json_data(self, data):
        """
        Sanitize data to ensure it's JSON serializable
        
        Args:
            data: Data to sanitize
            
        Returns:
            dict or list: Sanitized data that can be safely serialized to JSON
        """
        if isinstance(data, dict):
            # Process dictionary
            result = {}
            for key, value in data.items():
                # Skip keys with None values
                if value is None:
                    continue
                    
                # Recursively sanitize the value
                result[key] = self._sanitize_json_data(value)
            return result
        elif isinstance(data, list):
            # Process list items
            return [self._sanitize_json_data(item) for item in data if item is not None]
        elif isinstance(data, (str, int, float, bool)):
            # Basic JSON types are fine
            return data
        else:
            # Convert other types to strings
            try:
                return str(data)
            except:
                return None
            
    def _read_json(self, file_path):
        """
        Read data from a JSON file
        
        Args:
            file_path (str): Path to JSON file
            
        Returns:
            dict: JSON data
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Use strict=False to handle control characters
                return json.loads(f.read().strip())
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {file_path}: {str(e)}")
            # Attempt recovery by parsing only the valid part of the JSON
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                # Find the last valid JSON by looking for the last closing brace
                if content.rfind('}') > 0:
                    truncated = content[:content.rfind('}')+1]
                    return json.loads(truncated)
                return {}
            except:
                logger.error(f"Recovery attempt failed for {file_path}")
                return {}
        except Exception as e:
            logger.error(f"Error reading {file_path}: {str(e)}")
            return {}
            
    def _update_index(self, post_data):
        """
        Update the index with post information
        
        Args:
            post_data (dict): Post data
        """
        try:
            index_data = self._read_json(self.index_file)
            
            # Check if index_data is empty or invalid
            if not isinstance(index_data, dict):
                logger.warning(f"Index file contains invalid data, reinitializing")
                index_data = {"posts": {}, "last_updated": int(time.time())}
            
            if "posts" not in index_data:
                index_data["posts"] = {}
            
            # Update post entry
            post_id = post_data["id"]
            if post_id not in index_data["posts"]:
                index_data["posts"][post_id] = {}
                
            index_data["posts"][post_id].update({
                "subreddit": post_data["subreddit"],
                "title": post_data["title"],
                "author": post_data["author"],
                "created_utc": post_data["created_utc"],
                "collected_at": post_data["collected_at"]
            })
            
            # Update last_updated timestamp
            index_data["last_updated"] = int(time.time())
            
            # Write updated index
            self._write_json(self.index_file, index_data)
        except Exception as e:
            logger.error(f"Error in _update_index: {str(e)}")
        
    def _update_index_score(self, post_id, score):
        """
        Update the index with post score
        
        Args:
            post_id (str): Post ID
            score (float): Relevance score
        """
        try:
            index_data = self._read_json(self.index_file)
            
            # Check if index_data is empty or invalid
            if not isinstance(index_data, dict):
                logger.warning(f"Index file contains invalid data, reinitializing")
                index_data = {"posts": {}, "last_updated": int(time.time())}
            
            if "posts" not in index_data:
                index_data["posts"] = {}
            
            # Update post score
            if post_id in index_data["posts"]:
                index_data["posts"][post_id]["score"] = score
            else:
                # Create entry if it doesn't exist
                index_data["posts"][post_id] = {"score": score}
                
            # Update last_updated timestamp
            index_data["last_updated"] = int(time.time())
            
            # Write updated index
            self._write_json(self.index_file, index_data)
        except Exception as e:
            logger.error(f"Error in _update_index_score: {str(e)}")
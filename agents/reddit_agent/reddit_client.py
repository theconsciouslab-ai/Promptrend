# reddit_client.py
import praw
import time
import logging
from agents.reddit_agent.reddit_config import (
    REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT,
    POSTS_LIMIT
)


logger = logging.getLogger(__name__)

class RedditClient:
    """Client for interacting with the Reddit API"""
    
    def __init__(self):
        """Initialize the Reddit client with API credentials"""
        self.reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )
        logger.info("Reddit client initialized")
        
    def get_submissions(self, subreddit_name, sort_types=["new", "hot", "rising","top_day","top_week"]):
        """
        Get submissions from a subreddit
        
        Args:
            subreddit_name (str): Name of the subreddit
            sort_types (list): List of sorting types to collect
            
        Returns:
            list: List of submission objects
        """
        all_submissions = []
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            for sort_type in sort_types:
                logger.info(f"Collecting {sort_type} submissions from r/{subreddit_name}")
                
                # Handle different sort types
                if sort_type == "new":
                    submissions = subreddit.new(limit=POSTS_LIMIT)
                elif sort_type == "hot":
                    submissions = subreddit.hot(limit=POSTS_LIMIT)
                elif sort_type == "rising":
                    submissions = subreddit.rising(limit=POSTS_LIMIT)
                elif sort_type == "top_day":
                    submissions = subreddit.top("day", limit=POSTS_LIMIT)
                elif sort_type == "top_week":
                    submissions = subreddit.top("week", limit=POSTS_LIMIT)
                else:
                    logger.warning(f"Unknown sort type: {sort_type}, skipping")
                    continue
                
                # Convert to list and add to collection
                sort_submissions = list(submissions)
                logger.info(f"Collected {len(sort_submissions)} {sort_type} submissions from r/{subreddit_name}")
                all_submissions.extend(sort_submissions)
                
                # Sleep to respect rate limits
                time.sleep(2)
                
        except Exception as e:
            logger.error(f"Error collecting submissions from r/{subreddit_name}: {str(e)}")
            
        return all_submissions
    
    def get_comments(self, submission, depth=3):
        """
        Get comment tree for a submission
        
        Args:
            submission: Reddit submission object
            depth (int): Maximum depth to traverse in comment tree
            
        Returns:
            dict: Dictionary with comment tree structure
        """
        comment_tree = {}
        
        try:
            # Ensure all comments are loaded
            submission.comments.replace_more(limit=5)
            
            # Process comments recursively
            comment_tree = self._process_comments(submission.comments.list(), depth)
            logger.info(f"Collected {len(comment_tree)} comments for submission {submission.id}")
            
        except Exception as e:
            logger.error(f"Error collecting comments for submission {submission.id}: {str(e)}")
            
        return comment_tree
    
    def _process_comments(self, comments, max_depth, current_depth=0):
        """
        Recursively process comments up to max_depth
        
        Args:
            comments: List of comments
            max_depth (int): Maximum depth to traverse
            current_depth (int): Current depth in traversal
            
        Returns:
            dict: Processed comment tree
        """
        comment_tree = {}
        
        if current_depth > max_depth:
            return comment_tree
        
        for comment in comments:
            # Skip deleted/removed comments
            if comment.author is None or comment.body in ["[deleted]", "[removed]"]:
                continue
                
            # Add comment to tree
            comment_data = {
                "id": comment.id,
                "author": comment.author.name if comment.author else "[deleted]",
                "body": comment.body,
                "score": comment.score,
                "created_utc": comment.created_utc,
                "permalink": comment.permalink,
                "replies": {}
            }
            
            # Process replies if within depth limit
            if current_depth < max_depth and hasattr(comment, "replies"):
                replies = comment.replies.list()
                comment_data["replies"] = self._process_comments(
                    replies, max_depth, current_depth + 1
                )
                
            comment_tree[comment.id] = comment_data
            
        return comment_tree
# json_storage.py
import json
import os
import time
import logging
from pathlib import Path
from agents.twitter_agent.twitter_config import DATA_DIR

logger = logging.getLogger(__name__)

class JSONStorage:
    """Storage manager for saving Twitter data and analysis results as JSON files"""
    
    def __init__(self, data_dir=DATA_DIR):
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
            from agents.twitter_agent.twitter_config import KEYWORD_LEXICON
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
        from agents.twitter_agent.twitter_config import KEYWORD_LEXICON
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
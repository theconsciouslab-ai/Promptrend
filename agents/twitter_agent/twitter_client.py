# twitter_client.py
import logging
import tweepy
import time
from agents.twitter_agent.twitter_config import (
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
            
        tweet_id = tweet["id"] if isinstance(tweet, dict) else tweet.id
        conversation = {tweet_id: tweet}
        
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
        
        
    def handle_rate_limit(self):
        """
        Handle rate limits by sleeping for a specified duration
        """
        logger.warning("Rate limit hit. Sleeping for 60 seconds...")
        time.sleep(60)
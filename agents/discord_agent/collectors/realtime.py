# collectors/realtime.py
import asyncio
import logging
import discord
from datetime import datetime
import time
from agents.discord_agent.processors.message_processor import MessageProcessor
from agents.discord_agent.discord_config import (
    BOT_TOKEN,
TARGET_SERVERS,
LOG_FILE,
LOG_LEVEL,
DATA_DIR,
DEFAULT_REQUESTS_PER_SECOND,
DISCUSSION_SCORE_WEIGHT,
CODE_SCORE_WEIGHT,
ENABLE_PROMPT_EXTRACTION,
MONITOR_ALL_CHANNELS,
SECURITY_KEYWORDS,
TECHNICAL_KEYWORDS,
MODEL_KEYWORDS,
VULNERABILITY_KEYWORDS,
MIN_INLINE_CODE_LENGTH,
RELEVANT_URL_DOMAINS,
CODE_FILE_EXTENSIONS,
TEXT_FILE_EXTENSIONS,
COMMAND_PREFIX,
LDA_NUM_TOPICS,
LDA_MAX_DF,
LDA_MIN_DF,
LDA_TOPIC_MATCH_THRESHOLD,
LDA_TOPIC_SCORE_THRESHOLD,
CATEGORY_CONFIDENCE,
DEBUG_MODE,
RELEVANCE_THRESHOLD,
CONTEXT_WINDOW_SECONDS,
CONVERSATION_CACHE_TIMEOUT,
HISTORICAL_BACKFILL_DAYS,
HISTORICAL_BATCH_DELAY,
DISCORD_HISTORY_BATCH_SIZE,
LLM_MODEL,
LLM_PROVIDER,
MAX_RATE_LIMIT,
AZURE_ENDPOINT,
AZURE_API_VERSION,
AZURE_API_KEY,
LLM_MAX_RETRIES,
LLM_MAX_TOKENS,
LLM_TEMPERATURE,
LLM_SYSTEM_PROMPT,
)
logger = logging.getLogger("RealtimeCollector")

class RealtimeCollector:
    """
    Handles real-time monitoring of Discord channels.
    
    This collector processes new messages as they arrive and maintains
    conversation context for thread-based analysis.
    """
    
    def __init__(self, agent):
        """
        Initialize the realtime collector.
        
        Args:
            agent: The Discord agent instance
        """
        self.agent = agent
        self.message_processor = MessageProcessor()
        self.db_manager = agent.db_manager
        self.active_channels = set()
        self.conversation_cache = {}  # channel_id -> list of recent messages
        self.cache_timeout = CONVERSATION_CACHE_TIMEOUT  # seconds
    
    async def register_channel(self, channel):
        """
        Register a channel for real-time monitoring.
        
        Args:
            channel: Discord channel to monitor
        """
        self.active_channels.add(channel.id)
        logger.info(f"Registered channel for real-time monitoring: {channel.name}")
    
    async def process_message(self, message):
        """
        Process a new incoming message.
        
        Args:
            message: Discord message object
        """
        if message.channel.id not in self.active_channels:
            return
        
        print(f"Processing message in channel {message.channel.name}: {message.content}")
        print(f"[DEBUG] New message received in {message.channel.name}: {message.content}")
        
            
        # Update conversation cache
        self._update_conversation_cache(message)
        
        # Get message context (nearby messages in the same conversation)
        context = self._get_message_context(message)
        
        # Process the message with its context
        result = await self.message_processor.process_message_group(context, message.channel)
        
        # Store any detected vulnerabilities
        if result and result.get('is_vulnerability', False):
            self.db_manager.store_vulnerability(result)
            self.agent.stats["vulnerabilities_found"] += 1
        
        self.agent.stats["messages_processed"] += 1
    
    def _update_conversation_cache(self, message):
        """
        Update the conversation cache with a new message.
        
        Args:
            message: Discord message object
        """
        channel_id = message.channel.id
        
        # Initialize cache for this channel if needed
        if channel_id not in self.conversation_cache:
            self.conversation_cache[channel_id] = []
        
        # Add the new message
        self.conversation_cache[channel_id].append({
            'message': message,
            'timestamp': time.time()
        })
        
        # Clean up old messages
        self._cleanup_cache(channel_id)
    
    def _cleanup_cache(self, channel_id):
        """
        Remove old messages from the conversation cache.
        
        Args:
            channel_id: ID of the channel to clean up
        """
        if channel_id not in self.conversation_cache:
            return
            
        current_time = time.time()
        self.conversation_cache[channel_id] = [
            item for item in self.conversation_cache[channel_id]
            if current_time - item['timestamp'] < self.cache_timeout
        ]
    
    def _get_message_context(self, message):
        """
        Get contextual messages related to the current message.
        
        Args:
            message: Discord message object
            
        Returns:
            list: The message and its context as a conversation group
        """
        channel_id = message.channel.id
        
        # If message is in a thread, get the thread
        if hasattr(message, 'thread') and message.thread:
            # We would need to fetch thread messages here
            # For simplicity, we'll use cached messages
            thread_id = message.thread.id
            # This is simplified - in a real implementation, we'd fetch thread messages
            return [message]
        
        # If message is a reply, build reply chain
        if hasattr(message, 'reference') and message.reference:
            # This is simplified - in a real implementation, we'd fetch reply chain
            return [message]
        
        # Otherwise, use time-based context from cache
        if channel_id in self.conversation_cache:
            # Get messages from the last few minutes
            context_messages = [
                item['message'] for item in self.conversation_cache[channel_id]
                if abs((message.created_at.timestamp() -
                        item['message'].created_at.timestamp())) < CONTEXT_WINDOW_SECONDS

            ]
            return context_messages
        
        # Fallback to just the message itself
        return [message]
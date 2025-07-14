# collectors/historical.py
import asyncio
import logging
import discord
from datetime import datetime, timedelta
import aiohttp
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
logger = logging.getLogger("HistoricalCollector")

class HistoricalCollector:
    """
    Collects historical data from newly discovered Discord channels.
    
    This class handles the backfilling of message history for channels
    that have been newly added to the monitoring system.
    """
    
    def __init__(self, agent):
        """
        Initialize the historical collector.
        
        Args:
            agent: The Discord agent instance
        """
        self.agent = agent
        self.message_processor = MessageProcessor()
        self.backfill_days = HISTORICAL_BACKFILL_DAYS
        self.rate_limiter = agent.rate_limiter
        self.db_manager = agent.db_manager
        self.storage_manager = agent.storage_manager
    
    async def collect(self, channel):
        """
        Collect historical messages from a channel.
        
        Args:
            channel: Discord channel object to collect from
        """
        logger.info(f"Starting historical collection for channel: {channel.name} (ID: {channel.id})")
        
        # Calculate the cutoff date for backfilling
        cutoff_date = datetime.utcnow() - timedelta(days=self.backfill_days)
        
        try:
            # Retrieve messages in batches to respect rate limits
            total_processed = 0
            batch_size = DISCORD_HISTORY_BATCH_SIZE

            
            # Start with the most recent messages
            messages = []
            async for message in channel.history(limit=batch_size, after=cutoff_date):
                # Wait for rate limit if needed
                await self.rate_limiter.wait_if_needed()
                messages.append(message)
                
                # Process in batches to avoid too much memory usage
                if len(messages) >= batch_size:
                    await self._process_message_batch(messages, channel)
                    total_processed += len(messages)
                    messages = []
                    
                    # Small delay to not overwhelm Discord API
                    await asyncio.sleep(HISTORICAL_BATCH_DELAY)

            
            # Process any remaining messages
            if messages:
                await self._process_message_batch(messages, channel)
                total_processed += len(messages)
            
            logger.info(f"Completed historical collection for {channel.name}: processed {total_processed} messages")
            
            # Mark channel as known in storage
            self.storage_manager.mark_channel_known(channel.id)
            
            # Remove from newly discovered set
            if channel.id in self.agent.newly_discovered:
                self.agent.newly_discovered.remove(channel.id)
                
        except discord.errors.Forbidden:
            logger.error(f"No permission to read message history in channel: {channel.name}")
        except Exception as e:
            logger.error(f"Error during historical collection for {channel.name}: {str(e)}")
    
    async def _process_message_batch(self, messages, channel):
        """
        Process a batch of historical messages.
        
        Args:
            messages: List of message objects
            channel: The channel these messages are from
        """
        # Group messages into threads/conversations
        threaded_messages = self._group_message_threads(messages)
        
        # Process each thread
        for thread in threaded_messages:
            # Verify thread structure and relevance
            if not thread:
                continue
                
            # Process the thread
            result = await self.message_processor.process_message_group(thread, channel)
            
            # Store any detected vulnerabilities
            if result and result.get('is_vulnerability', False):
                self.storage_manager.store_vulnerability(result)
                self.agent.stats["vulnerabilities_found"] += 1
    
    def _group_message_threads(self, messages):
        """
        Group messages into conversation threads.
        
        Args:
            messages: List of message objects
            
        Returns:
            list: List of message groups representing conversations
        """
        # Sort messages by timestamp
        sorted_messages = sorted(messages, key=lambda m: m.created_at)
        
        # Group by Discord threads if available
        thread_groups = {}
        standalone_messages = []
        
        for message in sorted_messages:
            # For messages in threads
            if hasattr(message, 'thread') and message.thread:
                thread_id = message.thread.id
                if thread_id not in thread_groups:
                    thread_groups[thread_id] = []
                thread_groups[thread_id].append(message)
            # For reply chains
            elif hasattr(message, 'reference') and message.reference:
                ref_id = str(message.reference.message_id)
                if ref_id not in thread_groups:
                    thread_groups[ref_id] = []
                thread_groups[ref_id].append(message)
            # Standalone messages
            else:
                standalone_messages.append(message)
        
        # Convert dictionary to list of threads
        threads = list(thread_groups.values())
        
        # Add standalone messages as individual "threads"
        for msg in standalone_messages:
            threads.append([msg])
        
        return threads
# agent.py
import asyncio
import logging
import discord
from discord.ext import commands
import time
from datetime import datetime, timedelta

from agents.discord_agent.channel_classifier import ChannelClassifier
from agents.discord_agent.collectors.historical import HistoricalCollector
from agents.discord_agent.collectors.realtime import RealtimeCollector
from agents.discord_agent.processors.message_processor import MessageProcessor
from agents.discord_agent.utils.rate_limiter import RateLimiter
from agents.discord_agent.utils.storage_manager import StorageManager
from agents.discord_agent.discord_config import (
TARGET_SERVERS,
LOG_FILE,
LOG_LEVEL,
COMMAND_PREFIX,
)
# Configure logging
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("DiscordAgent")

class DiscordAgent(commands.Bot):
    """
    Main Discord Agent class for PrompTrend LLM vulnerability monitoring.
    
    This agent monitors Discord servers for LLM vulnerability-related discussions
    using a combination of channel categorization, temporal collection strategies,
    and LLM-based content analysis.
    """
    
    def __init__(self):
        """Initialize the Discord agent with necessary components."""
        # Initialize Discord bot with required intents
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        super().__init__(command_prefix=COMMAND_PREFIX, intents=intents)
        
        # Initialize components
        self.rate_limiter = RateLimiter()
        self.storage_manager = StorageManager()
        self.db_manager = self.storage_manager  # Temporary alias to avoid errors

        self.channel_classifier = ChannelClassifier()
        self.message_processor = MessageProcessor()
        
        # Collectors
        self.historical_collector = HistoricalCollector(self)
        self.realtime_collector = RealtimeCollector(self)
        
        # State tracking
        self.monitored_channels = {}  # guild_id -> [channel_ids]  
        self.monitored_servers = TARGET_SERVERS
        self.newly_discovered = set()  # Set of newly discovered channels
        
        # Statistics
        self.stats = {
            "messages_processed": 0,
            "vulnerabilities_found": 0,
            "channels_monitored": 0
        }
        
    async def setup_hook(self):
        """Setup tasks to run when the bot connects."""
        self.loop.create_task(self.initialize_monitoring())
    
    async def on_ready(self):
        """Called when the bot has connected to Discord."""
        logger.info(f"Logged in as {self.user.name} (ID: {self.user.id})")
        logger.info(f"Monitoring {len(self.monitored_servers)} servers")
        
    async def initialize_monitoring(self):
        """Initialize the monitoring process for all target servers."""
        await self.wait_until_ready()
        
        logger.info("Beginning server and channel classification...")
        guild_count = 0
        
        for guild in self.guilds:
            # Skip guilds not in our monitored list
            if guild.name not in self.monitored_servers and guild.id not in self.monitored_servers:
                continue
                
            guild_count += 1
            logger.info(f"Processing guild: {guild.name} (ID: {guild.id})")
            
            # Classify and get ALL channels to monitor
            monitored_channels = await self.channel_classifier.classify_channels(guild)
            self.monitored_channels[guild.id] = monitored_channels
            
            # Track newly discovered channels (those not in our storage)
            for channel_id in monitored_channels:
                channel = guild.get_channel(channel_id)
                if channel and not self.storage_manager.is_channel_known(channel_id):
                    self.newly_discovered.add(channel_id)
            
            # Start monitoring ALL channels
            await self.begin_monitoring(guild, monitored_channels)
            
        self.stats["channels_monitored"] = sum(len(channels) for channels in self.monitored_channels.values())
        logger.info(f"Initialized monitoring for {guild_count} guilds and {self.stats['channels_monitored']} channels")
    
    async def begin_monitoring(self, guild, channel_ids):
        """Start monitoring the channels in a guild."""
        for channel_id in channel_ids:
            channel = guild.get_channel(channel_id)
            if not channel:
                continue
                
            # Historical collection for newly discovered channels
            if channel_id in self.newly_discovered:
                logger.info(f"Starting historical collection for new channel: {channel.name} (ID: {channel_id})")
                await self.historical_collector.collect(channel)
                
            # Register for real-time monitoring
            await self.realtime_collector.register_channel(channel)
    
    async def on_message(self, message):
        """Handle incoming messages for real-time monitoring."""
        # Ignore bot messages
        if message.author.bot:
            return
            
        # Only process messages in monitored channels
        guild_id = message.guild.id if message.guild else None
        if (not guild_id or 
            guild_id not in self.monitored_channels or 
            message.channel.id not in self.monitored_channels[guild_id]):
            return
        
        # Process the message
        await self.realtime_collector.process_message(message)
        
    async def close(self):
        """Clean up resources when shutting down."""
        # Save any pending data
        self.storage_manager.close()
        await super().close()
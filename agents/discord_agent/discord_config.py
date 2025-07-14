# config.py
"""
Configuration settings for the Discord Agent.
"""

import os
from dotenv import load_dotenv
import logging

load_dotenv()  # Load from .env file

# Discord Bot Settings
BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

# Target Discord Servers
# Can be server names or server IDs
TARGET_SERVERS = [
    #"SERVER_NAME_1",
    #"SERVER_NAME_2",
]

# Logging Settings
LOG_FILE = "logs/discord_agent.log"
LOG_LEVEL = logging.INFO

DATA_DIR =  os.path.join("Data", "Discord_Data")

# Rate Limiting Settings
DEFAULT_REQUESTS_PER_SECOND = 1

# Scoring Weights for Discussion and Code
DISCUSSION_SCORE_WEIGHT = 0.4
CODE_SCORE_WEIGHT = 0.4

#  Prompt Extraction Settings
ENABLE_PROMPT_EXTRACTION = True


# Monitor Settings
MONITOR_ALL_CHANNELS = True  # Set to True to monitor ALL channels

# Keyword Lexicons for Classification and Filtering
SECURITY_KEYWORDS = {
    "vulnerability", "exploit", "attack", "hack", "security",
    "prompt injection", "jailbreak", "penetration testing", "red team",
    "bypass", "security flaw", "vulnerability disclosure"
}

TECHNICAL_KEYWORDS = {
    "code", "programming", "python", "javascript", "api",
    "implementation", "algorithm", "development", "llm", "gpt",
    "claude", "neural network", "machine learning", "ai", "fine-tuning"
}

MODEL_KEYWORDS = {
    "gpt", "gpt-4", "gpt-3", "claude", "llama", "falcon", "mistral",
    "large language model", "transformer", "openai", "anthropic",
    "bert", "hugging face", "embedding", "diffusion"
}

VULNERABILITY_KEYWORDS = {
    # General vulnerability terms
    "vulnerability", "exploit", "attack", "bypass", "injection",
    "jailbreak", "security flaw", "hack", "compromise",
    
    # LLM-specific terms
    "prompt injection", "prompt leaking", "indirect prompt injection",
    "data extraction", "model inversion", "sycophant", "hallucination",
    "training data extraction", "prompt bypass", "instruction override",
    "system prompt", "model extraction", "adversarial prompt", 
    "security boundary", "model poisoning", "backdoor",
    
    # LLM-specific attack techniques
    "DAN", "Do Anything Now", "jail break", "grandma attack",
    "token smuggling", "unicode exploit", "suffix injection",
    "prefix injection", "context manipulation", "system prompt leak",
    
    # Known frameworks/tools
    "GCG", "AutoDAN", "Red-Team", "PAIR", "HackLLM",
    "DeepInception", "RAUGH", "Gandalf", "jailbreakchant",
    
    # Technical terms
    "token", "embedding", "context window", "parser", "sanitization",
    "validation", "safety filter", "content moderation", "guardrail"
}

MIN_INLINE_CODE_LENGTH = 30

RELEVANT_URL_DOMAINS = {
    'github.com', 'gitlab.com', 'bitbucket.org',
    'huggingface.co', 'arxiv.org', 'openai.com',
    'anthropic.com', 'tensorflow.org', 'pytorch.org',
    'stackoverflow.com', 'gist.github.com'
}

CODE_FILE_EXTENSIONS = {
    '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.rb', '.php',
    '.go', '.rs', '.c', '.cpp', '.h', '.cs', '.sh', '.ps1', '.ipynb',
    '.json', '.yaml', '.yml', '.toml', '.xml', '.html', '.css'
}

TEXT_FILE_EXTENSIONS = {'.txt', '.md', '.csv', '.log'}


# Command Settings
COMMAND_PREFIX = "!"

# --- LDA Topic Modeling ---
LDA_NUM_TOPICS = 5
LDA_MAX_DF = 0.95
LDA_MIN_DF = 2
LDA_TOPIC_MATCH_THRESHOLD = 0.3  # Minimum topic relevance to accept
LDA_TOPIC_SCORE_THRESHOLD = 0.1  # Minimum match score to keep a topic-category link

# --- Category Scoring ---
CATEGORY_CONFIDENCE = {
    "security": 0.9,
    "technical": 0.8,
    "model_specific": 0.7,
    "general": 0.5
}

# --- Temporal Collection ---
CONTEXT_WINDOW_SECONDS = 300  # Used to fetch nearby messages for context

# --- Historical Collection ---
DISCORD_HISTORY_BATCH_SIZE = 100
HISTORICAL_BATCH_DELAY = 2.0  # Seconds between historical batches


# Collection Settings
HISTORICAL_BACKFILL_DAYS = 30  # Days to backfill for new channels
CONVERSATION_CACHE_TIMEOUT = 1800  # Seconds to keep messages in conversation cache (30 minutes)

# Analysis Settings
RELEVANCE_THRESHOLD = 0.1  # Minimum score to consider a vulnerability
DEBUG_MODE = True  # Enable debug mode for detailed logging
MAX_RATE_LIMIT = 50  # Maximum rate limit for API calls (requests per second)

# LLM API Settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER")
LLM_MODEL = os.getenv("LLM_MODEL")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
# LLM Settings
LLM_MAX_RETRIES = 3
LLM_MAX_TOKENS = 10
LLM_TEMPERATURE = 0.1
LLM_SYSTEM_PROMPT = "You are an LLM security evaluator. Respond with a float score only."

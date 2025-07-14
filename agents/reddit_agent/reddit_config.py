# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# JSON Storage Configuration
DATA_DIR =  os.path.join("Data", "Reddit_Data")

# Reddit API Configuration
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "RedditLLMSecurityAgent/1.0")

# LLM API Configuration
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-05-01-preview")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "azure")
LLM_MODEL = os.getenv("LLM_MODEL", "Deepseek-V3")

# Agent Configuration
TARGET_SUBREDDITS = [
    "ChatGPTJailbreak",
    "ChatGptDAN",
    "PromptEngineering",
    "LocalLLaMA",
    "ArtificialInteligence",
    "ChatGPT",
    "LLMDevs",
    "AI_Agents",
    "MachineLearning",
    "cybersecurity",
    "netsec",
    "hacking",
    "GPT_jailbreaks",
    "LanguageTechnology",
    "singularity",
    "ChatGPTPromptGenius",
    "LangChain",
    "ChatGPTCoding",
    "HowToHack"
    
]

# Collection parameters
POSTS_LIMIT = int(
    os.getenv("POSTS_LIMIT", 25)
)  # Number of posts to collect per subreddit
COMMENT_DEPTH = int(os.getenv("COMMENT_DEPTH", 3))  # Depth of comment tree to traverse
COLLECTION_INTERVAL = int(
    os.getenv("COLLECTION_INTERVAL", 1800)
)  # Collection frequency in seconds

# Filtering parameters
KEYWORD_RELEVANCE_THRESHOLD = float(os.getenv("KEYWORD_RELEVANCE_THRESHOLD", 0.3))
LLM_RELEVANCE_THRESHOLD = float(os.getenv("LLM_RELEVANCE_THRESHOLD", 0.4))

# Initial keyword lexicon
KEYWORD_LEXICON = [
    "jailbreak",
    "DAN",
    "vulnerability",
    "bypass",
    "prompt injection",
    "safety",
    "alignment",
    "exploit",
    "hack",
    "red team",
    "red teaming",
    "security",
    "circumvent",
    "workaround",
    "backdoor",
    "attack",
    "hallucination",
    "extraction",
    "system prompt",
    "leaking",
    "RLHF",
    "data poisoning",
    "model poisoning",
    "adversarial",
    "sandbox",
    "escape",
    "authentication",
    "unauthorized access",
    "unlock",
    "access",
    "bypass",
    "exploit",
    "malicious",
    "unfiltered",
    "unmoderated",
    "unrestricted",
    "NSFW",
    "bypassed",
    "unbanned",
    "unfiltered",
    "response restrictions",
    "instructions/prompt",
]


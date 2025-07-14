# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# === Storage Path ===
DATA_DIR =  os.path.join("Data", "Twitter_Data")

# === Twitter API Configuration ===
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")

# === Azure OpenAI Configuration ===
# LLM API Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "azure")
LLM_MODEL = os.getenv("LLM_MODEL", "DeepSeek-V3")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2023-05-15")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
# === Agent Configuration ===
SEED_USERS = [
    "elder_plinius",
]

USER_TWEETS_LIMIT = int(os.getenv("USER_TWEETS_LIMIT", 100))
MAX_USERS = int(os.getenv("MAX_USERS", 2))
CONVERSATION_DEPTH = int(os.getenv("CONVERSATION_DEPTH", 5))
COLLECTION_INTERVAL = int(os.getenv("COLLECTION_INTERVAL", 3600))

# === Filtering Parameters ===
KEYWORD_RELEVANCE_THRESHOLD = float(os.getenv("KEYWORD_RELEVANCE_THRESHOLD", 0.1))
LLM_RELEVANCE_THRESHOLD = float(os.getenv("LLM_RELEVANCE_THRESHOLD", 0.1))

# === Keyword Lexicon ===
KEYWORD_LEXICON = [
    "jailbreak", "DAN", "vulnerability", "bypass", "prompt injection",
    "safety", "alignment", "exploit", "hack", "red team", "red teaming",
    "security", "circumvent", "workaround", "backdoor", "attack",
    "hallucination", "extraction", "system prompt", "leaking",
    "RLHF", "data poisoning", "model poisoning", "adversarial",
    "sandbox", "escape", "authentication", "unauthorized access"
]


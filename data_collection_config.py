# config.py
import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

"""
Configuration settings for the PrompTrend system.
"""

# Agent configurations
REGIONAL_CONFIGS = [
    {
        "id": "region-1",
        "platforms": ["reddit", "forum"]
    },
    {
        "id": "region-2",
        "platforms": ["discord", "twitter"]
    },
    {
       "id": "region-3",
       "platforms": ["github"]
    },
   
]



# -----------------------------------------------------------------------------
# List of all platforms (agents) to run – used by MasterController
PLATFORMS = [
    platform
    for region in REGIONAL_CONFIGS
    for platform in region["platforms"]
]


# List of agent names whose adapters live under agents/<name>_agent/
PLATFORMS = [
    "reddit",
    "twitter",
    "discord",
    "github",
    "forum",
]

# Lexicon configuration
LEXICON = {
    # Core terminology from established taxonomies
    "core_terms": [
        "prompt injection",
        "jailbreak",
        "DAN",
        "system prompt",
        "bypass",
        "data extraction",
        "model inversion",
        "prompt leaking",
        "system instruction",
        "token smuggling",
        "adversarial prompt"
    ],
    
    # Contextual patterns (patterns that suggest exploits)
    "contextual_patterns": {
        r"ignore\s+(?:previous|all)\s+(?:instructions|prompts)": 0.15,
        r"bypass.*(?:filter|restriction|limitation)": 0.15,
        r"(?:pretend|roleplay|imagine)\s+you\s+are": 0.10,
        r"let's\s+play\s+a\s+game": 0.10,
        r"I\s+(?:need|want)\s+you\s+to\s+(?:pretend|act)": 0.15,
        r"do\s+(?:not|n't)\s+(?:consider|think about|remember)": 0.15,
        r"exploit": 0.15,
        r"vulnerability": 0.10,
        r"attack": 0.10
    },
    
    # Emergent terminology (dynamically updated)
    "emergent_terms": [
        "grandma attack",
        "sycophant",
        "GPT-4 Simulator",
        "ReAct",
        "AutoGPT",
        "model extraction",
        "DALL-E Prompt",
        "Sandbox escape",
        "base64",
        "Unicode exploit",
        "prefix injection",
        "suffix injection"
    ],
    
    # Threshold for adding new terms
    "emergence_threshold": 0.6
}

# Platform-specific relevance factors
PLATFORM_RELEVANCE_FACTORS = {
    "reddit": 1.0,
    "twitter": 0.9,
    "discord": 1.2,  # Discord often has more technical discussions
    "github": 1.3,   # GitHub contains more implementation details
    "forum": 1.1     # Specialized forums often have deeper discussions
}

# Vulnerability threshold
VULNERABILITY_THRESHOLD = 0.5

# Privacy configuration
PRIVACY_CONFIG = {
    "hash_salt": "PrompTrendRandomSalt123456789",  # Change in production
    "anonymize_usernames": True,
    "sanitize_text": True,
    "redact_pii": True
}

# Relationship thresholds
RELATIONSHIP_THRESHOLDS = {
    "variant": 0.2,        # Threshold for variant relationship
    "cross_platform": 0.2,  # Threshold for cross-platform relationship
    "technical": 0.2       # Threshold for technical similarity
}

# Schema versions
SCHEMA_VERSIONS = {
    "raw": "1.0",
    "normalized": "1.0",
    "enriched": "1.0",
    "analytical": "1.0",
    "document": "1.0"
}

KEYWORD_LEXICON = [
    "jailbreak", "DAN", "vulnerability", "bypass", "prompt injection",
    "safety", "alignment", "exploit", "hack", "red team", "red teaming",
    "security", "circumvent", "workaround", "backdoor", "attack",
    "hallucination", "extraction", "system prompt", "leaking",
    "RLHF", "data poisoning", "model poisoning", "adversarial",
    "sandbox", "escape", "authentication", "unauthorized access"
]

# LLM configuration for content analysis
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-05-01-preview")
AZURE_MAX_REQUESTS_PER_MINUTE = int(os.getenv("AZURE_MAX_REQUESTS_PER_MINUTE", 20))
AZURE_MAX_RETRIES = int(os.getenv("AZURE_MAX_RETRIES", 5))
AZURE_CONCURRENCY = int(os.getenv("AZURE_CONCURRENCY", 3))
LLM_MODEL = os.getenv("LLM_MODEL", "Deepseek-V3")


# config.py
"""
Configuration settings for the Discussion Forums Agent.
"""
from typing import Dict, List, Any
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()


# Forum Collection Settings
TARGET_FORUMS = {
    "openai_community": {
        "name": "OpenAI Community",
        "url": "https://community.openai.com/",
        "platform_type": "discourse"
    },
    "netskope_ai": {
        "name": "Netskope AI & ML",
        "url": "https://community.netskope.com/artificial-intelligence-and-machine-learning-10",
        "platform_type": "custom"
    },
    "nvidia_dev": {
        "name": "NVIDIA Developer Forums",
        "url": "https://forums.developer.nvidia.com/",
        "platform_type": "discourse"
    },
    "isecur1ty": {
        "name": "iSecur1ty",
        "url": "https://isecur1ty.org/",
        "platform_type": "custom"
    },
    "arab_security": {
        "name": "Arab Security Conference",
        "url": "https://www.arabsecurityconference.com/",
        "platform_type": "custom"
    },
    "anquanke": {
        "name": "Anquanke",
        "url": "https://www.anquanke.com/",
        "platform_type": "custom"
    },
    "habr_infosec": {
        "name": "Habr Infosecurity",
        "url": "https://habr.com/ru/hub/infosecurity/",
        "platform_type": "custom"
    },
    "ciso_forum": {
        "name": "CISO Forum Russia",
        "url": "https://infosecurity-forum.ru/",
        "platform_type": "custom"
    },
    #"gis_days": {
     #   "name": "GIS DAYS",
     #   "url": "https://gis-days.ru/",
    #    "platform_type": "custom"
   # },
    
   
    "anakin_ai": {
        "name": "Anakin AI Jailbreak Blog",
        "url": "https://anakin.ai/blog/chatgpt-jailbreak-prompts/",
        "platform_type": "custom"
    },
    "strippedfilm": {
        "name": "StrippedFilm Jailbreak Guide",
        "url": "https://strippedfilm.com/chatgpt-jailbreak-prompts/",
        "platform_type": "custom"
    },
    "learn_prompting": {
        "name": "Learn Prompting - Injection",
        "url": "https://learnprompting.org/docs/prompt_hacking/injection",
        "platform_type": "custom"
    },
    
    "hacker_news": {
        "name": "Hacker News",
        "url": "https://news.ycombinator.com/",
        "platform_type": "custom"
    },
    "devtalk_prompt_injection": {
        "name": "Devtalk Prompt Injection Thread",
        "url": "https://forum.devtalk.com/t/prompt-injection-what-s-the-worst-that-can-happen/105030",
        "platform_type": "custom"
    }
}

DISCUSSION_FORUMS_CYCLE_SECONDS =  3600  # Default to 1 hour

# Directory for storing collected forum JSONs
FORUM_DATA_PATH = "data/Forum_data"


# Collection Settings
FORUM_LOOKBACK_DAYS = 30  # Days to look back for initial collection
FORUM_RELEVANCE_THRESHOLD = 0.1  # Minimum score to consider a thread relevant

# HTTP Request Settings
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "DNT": "1"
}
REQUEST_TIMEOUT = 30  # seconds
INTER_PAGE_DELAY = 3  # seconds between page requests
INTER_FORUM_DELAY = 10  # seconds between different forums

# IP Rotation Settings
USE_PROXY_ROTATION = False
PROXIES = [
    # Add your proxy list here
    "http://proxy1.example.com:8080",
    "http://proxy2.example.com:8080"
]
RATE_LIMIT_DELAY = 5  # seconds

# Language and Translation Settings
ENABLE_TRANSLATION = os.getenv("ENABLE_TRANSLATION", "true") == "true"
FASTTEXT_MODEL_PATH = os.getenv("FASTTEXT_MODEL_PATH")

TRANSLATION_API_TYPE = "deepl"  # "google" or "deepl"
GOOGLE_TRANSLATE_API_KEY = os.getenv("GOOGLE_TRANSLATE_API_KEY")
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")
# Cultural Analysis Settings
CULTURAL_WEIGHTS = {
    # Language-specific scoring weights
    "en": {
        "cultural": 0.2,
        "technical": 0.4,
        "llm": 0.4
    },
    "ar": {
        "cultural": 0.4,  # Higher cultural context weight for Arabic
        "technical": 0.3,
        "llm": 0.3
    },
    "zh": {
        "cultural": 0.4,  # Higher cultural context weight for Chinese
        "technical": 0.3,
        "llm": 0.3
    },
    "ru": {
        "cultural": 0.3,  # Medium cultural context weight for Russian
        "technical": 0.4,
        "llm": 0.3
    },
    "default": {
        "cultural": 0.2,
        "technical": 0.4,
        "llm": 0.4
    }
}

# Language-specific keywords for relevance detection
MULTILINGUAL_LEXICONS = {
    "en": [
        # English keywords
        "vulnerability", "exploit", "attack", "hack", "security",
        "jailbreak", "prompt injection", "red team", "bypass",
        "llm", "gpt", "claude", "llama", "language model",
        "chatgpt", "anthropic", "openai", "huggingface"
    ],
    "ar": [
        # Arabic keywords
        "ثغرة", "استغلال", "هجوم", "اختراق", "أمان",
        "تجاوز", "حقن", "اختبار اختراق", "فريق أحمر",
        "نموذج لغة كبير", "شات جي بي تي", "كلود", "لاما"
    ],
    "zh": [
        # Chinese keywords
        "漏洞", "利用", "攻击", "黑客", "安全",
        "越狱", "提示注入", "红队", "绕过",
        "大型语言模型", "聊天机器人", "人工智能"
    ],
    "ru": [
        # Russian keywords
        "уязвимость", "эксплойт", "атака", "взлом", "безопасность",
        "джейлбрейк", "инъекция промпта", "красная команда", "обход",
        "языковая модель", "чат-бот", "ИИ"
    ]
}

# LLM Prompts with cultural context
PROMPT_TEMPLATES = {
    "en": """
    Analyze if this forum thread from {forum_name} discusses LLM security vulnerabilities or exploits.
    
    TITLE: {title}
    
    CONTENT:
    {content}
    
    Rate the likelihood this thread contains LLM vulnerability information on a scale of 0.0 to 1.0, where:
    - 0.0 means no relation to LLM vulnerabilities
    - 0.5 means possibly related, with indirect references
    - 1.0 means definitely discusses LLM vulnerabilities or exploits
    
    Return only a single number between 0.0 and 1.0, with no explanation.
    """,
    
    "ar": """
    Analyze if this Arabic forum thread from {forum_name} discusses LLM security vulnerabilities or exploits.
    
    TITLE: {title}
    
    CONTENT:
    {content}
    
    Consider the cultural context: Arabic technical discussions often use metaphorical language and indirect references when discussing security vulnerabilities.
    
    Rate the likelihood this thread contains LLM vulnerability information on a scale of 0.0 to 1.0, where:
    - 0.0 means no relation to LLM vulnerabilities
    - 0.5 means possibly related, with indirect references
    - 1.0 means definitely discusses LLM vulnerabilities or exploits
    
    Return only a single number between 0.0 and 1.0, with no explanation.
    """,
    
    "zh": """
    Analyze if this Chinese forum thread from {forum_name} discusses LLM security vulnerabilities or exploits.
    
    TITLE: {title}
    
    CONTENT:
    {content}
    
    Consider the cultural context: Chinese technical discussions often use metaphorical language and indirect references when discussing security topics, and may refer to Western tools by different names.
    
    Rate the likelihood this thread contains LLM vulnerability information on a scale of 0.0 to 1.0, where:
    - 0.0 means no relation to LLM vulnerabilities
    - 0.5 means possibly related, with indirect references
    - 1.0 means definitely discusses LLM vulnerabilities or exploits
    
    Return only a single number between 0.0 and 1.0, with no explanation.
    """,
    
    "ru": """
    Analyze if this Russian forum thread from {forum_name} discusses LLM security vulnerabilities or exploits.
    
    TITLE: {title}
    
    CONTENT:
    {content}
    
    Consider the cultural context: Russian security forums often include technical details and code examples, and discussions about vulnerabilities may be more direct.
    
    Rate the likelihood this thread contains LLM vulnerability information on a scale of 0.0 to 1.0, where:
    - 0.0 means no relation to LLM vulnerabilities
    - 0.5 means possibly related, with indirect references
    - 1.0 means definitely discusses LLM vulnerabilities or exploits
    
    Return only a single number between 0.0 and 1.0, with no explanation.
    """,
    
    "default": """
    Analyze if this forum thread from {forum_name} discusses LLM security vulnerabilities or exploits.
    
    TITLE: {title}
    
    CONTENT:
    {content}
    
    Consider the language context: This content is in {language}.
    
    Rate the likelihood this thread contains LLM vulnerability information on a scale of 0.0 to 1.0, where:
    - 0.0 means no relation to LLM vulnerabilities
    - 0.5 means possibly related, with indirect references
    - 1.0 means definitely discusses LLM vulnerabilities or exploits
    
    Return only a single number between 0.0 and 1.0, with no explanation.
    """
}

# LLM API Settings
LLM_PROVIDER="azure"
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
LLM_MODEL = os.getenv("LLM_MODEL")
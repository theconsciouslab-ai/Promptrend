# config.py

import os
from dotenv import load_dotenv

load_dotenv()


"""

Configuration settings for the GitHub Agent.

"""


DATA_DIR =  os.path.join("Data", "GitHub_Data")
# GitHub API Settings

GITHUB_API_TOKEN = os.getenv("GITHUB_API_TOKEN")  # Replace with your actual token

# LLM API Settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "azure")  # Options: "azure", "openai", "local"
LLM_MODEL = os.getenv(
    "LLM_MODEL", "DeepSeek-V3"
)  # Options: "DeepSeek-V3", "gpt-4", "gpt-3.5-turbo", etc.
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")

AZURE_MAX_REQUESTS_PER_MINUTE = 20
AZURE_MAX_RETRIES = 3
AZURE_CONCURRENCY = 5  # Number of concurrent requests to Azure API
    

# Target GitHub Repositories
COLLECTION_INTERVAL =  3600 
# Format: "owner/repo"

TARGET_REPOSITORIES = [
    # Original repositories
    "hwchase17/langchain",
    "ggerganov/llama.cpp",
    "tatsu-lab/stanford_alpaca",
    "AUTOMATIC1111/stable-diffusion-webui",
    "lm-sys/FastChat",
    "Significant-Gravitas/Auto-GPT",
    "gpt-engineer-org/gpt-engineer",
    "microsoft/Security-101",
    "greshake/llm-security",
    "elder-plinius/CL4R1T4S",
    
    # Large-Scale Datasets & Primary Collections
    "verazuo/jailbreak_llms",
    "CyberAlbSecOP/Awesome_GPT_Super_Prompting",
    "yueliu1999/Awesome-Jailbreak-on-LLMs",
    "0xeb/TheBigPromptLibrary",
    
    # Security Research & Tools
    "PromptLabs/Prompt-Hacking-Resources",
    "corca-ai/awesome-llm-security",
    "sinanw/llm-security-prompt-injection",
    "leeisack/jailbreak_llm",
    "langgptai/LLM-Jailbreaks",
    
    # DAN & Classic Jailbreaks
    "0xk1h0/ChatGPT_DAN",
    "Cyberlion-Technologies/ChatGPT_DAN",
    "GabryB03/ChatGPT-Jailbreaks",
    "catsanzsh/01preview-jailbreaks",
    "yes133/ChatGPT-Prompts-Jailbreaks-And-More",
    "jackhhao/jailbreak-classification",
    
    # Detection & Defense Tools
    "deadbits/vigil",
    "protectai/rebuff",
    "laiyer-ai/llm-guard",
    
    # Research Papers & Analysis
    "ThuCCSLab/Awesome-LM-SSP",
    "microsoft/promptbench",
    
    # Specialized Datasets
    "compass-ctf-team/prompt_injection_research",
    "rabbidave/Denzel-Crocker-Hunting-For-Fairly-Odd-Prompts",
    "deadbits/vigil-jailbreak-ada-002",
    
    # System Prompt Leaks
    "LouisShark/chatgpt_system_prompt",
    "friuns2/Leaked-GPTs",
    "tjadamlee/GPTs-prompts",
    "linexjlin/GPTs",
    "B3o/GPTS-Prompt-Collection",
    "1003715231/gptstore-prompts",
    "adamidarrha/TopGptPrompts",
    "friuns2/BlackFriday-GPTs-Prompts",
    "parmarjh/Leaked-GPTs",
    
    # Implementation Tools & Collections
    "utkusen/promptmap",
    "yunwei37/prompt-hacker-collections",
    "THUDM/ChatGLM-6B",
    "jzzjackz/chatgptjailbreaks",
    "rubend18/ChatGPT-Jailbreak-Prompts",
    "tg12/gpt_jailbreak_status",
    
    # Additional Security Collections
    "Cranot/chatbot-injections-exploits",
    "FonduAI/awesome-prompt-injection",
    "TakSec/Prompt-Injection-Everywhere",
    "gogooing/Awesome-GPTs",
    "lxfater/Awesome-GPTs",
    "Superdev0909/Awesome-AI-GPTs-main",
    "SuperShinyDev/ChatGPTApplication",
    
    # Multi-modal & Advanced Attacks
    "cyberark/FuzzyAI",
    "AgentOps-AI/BestGPTs",
    "fr0gger/Awesome-GPT-Agents",
    "cckuailong/awesome-gpt-security",
    "LiLittleCat/awesome-free-chatgpt",
    "cheahjs/free-llm-api-resources",
    "sindresorhus/awesome-chatgpt",
    
    # Research & Educational
    "EmbraceAGI/Awesome-AI-GPTs",
    "Anil-matcha/Awesome-GPT-Store",
    "friuns2/Awesome-GPTs-Big-List",
]


# Vulnerability Patterns
CODE_VULNERABILITY_PATTERNS = [
    # Prompt injection patterns
    r'prompt\s*=\s*["\']system.*["\']',
    r"(?:user|assistant|system)_prompt\s*=",
    r"(?:bypass|avoid|trick).*(?:filter|moderation|safety)",
    # Jailbreak patterns
    r"jailbreak",
    r"DAN|do\s+anything\s+now",
    r"ignore\s+(?:previous|prior)\s+instructions",
    # Training data extraction
    r"extract\s+(?:training|private)\s+data",
    r"model\s+inversion",
    r"membership\s+inference",
    # Prompt manipulation
    r"prompt\s+(?:manipulation|injection|attack)",
    r"token\s+(?:smuggling|manipulation)",
    # Model exploitation
    r"exploit\s+model",
    r"adversarial\s+(?:example|input)",
    r"(?:red|adversarial)\s+team",
    r"adversarial\s+training",
    r"adversarial\s+attack",
    r"adversarial\s+example",
    r"adversarial\s+input",
    r"adversarial\s+prompt",
    r"adversarial\s+context",
    r"(?i)ignore\s+.*instructions",
    r"(?i)pretend\s+to\s+be",
    r"(?i)you\s+are\s+not\s+an\s+ai",
    r"(?i)chatgpt\s*[,:\-]?\s*break\s+free",
    r"(?i)do\s+anything\s+now",
    r"(?i)act\s+as\s+if",
    r"(?i)you\s+are\s+not\s",
    r"(?i)you\s+are\s+an\s+ai",
    r"(?i)you\s+are\s+not\s+an\s+ai",
    r"(?i)you\s+are\s+not\s",
]

# LLM API Patterns
CODE_API_PATTERNS = [
    r"openai\.Completion\.create",
    r"openai\.ChatCompletion\.create",
    r"anthropic\.Completion",
    r'completion\s*\(\s*model=["\']gpt',
    r"from\s+transformers\s+import",
    r"HuggingFaceHub",
    r"LangChain",
    r"generate_text",
    r"generate_response",
    r"llm\.[a-zA-Z_]+\(",
]




# Collection Settings

REPOSITORY_LOOKBACK_DAYS = 30  # Days to look back for initial collection

REPO_RELEVANCE_THRESHOLD = 0.3  # Minimum score to consider a repository relevant

CODE_RELEVANCE_THRESHOLD = 0.3  # Minimum score to consider code a vulnerability


# Lexicons for Classification and Filtering

SECURITY_KEYWORDS = [
    "vulnerability",
    "exploit",
    "attack",
    "hack",
    "security",
    "prompt injection",
    "jailbreak",
    "penetration testing",
    "red team",
    "bypass",
    "security flaw",
    "vulnerability disclosure",
    "security audit",
    "security assessment",
    "security analysis",
    "security research",
    "security testing",
    "security vulnerability",
    "security incident",
]

SECURITY_PATTERNS = [
    r"\bexploit\b",
    r"\bvulnerability\b",
    r"\battack\b",
    r"\bhack\b",
    r"\bsecurity\b",
    r"\bprompt injection\b",
    r"\bjailbreak\b",
    r"\bpenetration testing\b",
    r"\bred team\b",
    r"\bbypass\b",
    r"\bsecurity flaw\b",
    r"\bvulnerability disclosure\b",
    r"\bsecurity audit\b",
    r"\bsecurity assessment\b",
    r"\bsecurity analysis\b",
    r"(?:security|vulnerability)\s+(?:issue|report|bug|flaw)",
    r"(?:prompt|instruction)\s+(?:injection|manipulation|leak)",
    r"(?:bypass|evade|avoid)\s+(?:filter|moderation|safety)",
    r"exploit\s+(?:found|discovered|identified)",
    r"jailbreak\s+(?:technique|method|approach)",
    r"CVE-\d{4}-\d{4,}",
    r"[rR]esponsible\s+[dD]isclosure",
    r"[sS]ecurity\s+[rR]esearcher",
]


LLM_KEYWORDS = [
    "gpt",
    "gpt-4",
    "gpt-3",
    "claude",
    "llama",
    "falcon",
    "mistral",
    "large language model",
    "transformer",
    "openai",
    "anthropic",
    "bert",
    "hugging face",
    "embedding",
    "diffusion",
    "llm",
    "chatgpt",
    "gpt4",
    "language model",
    "text generation",
    "text-to-text",
    "text generation",
]


VULNERABILITY_KEYWORDS = [
    # General vulnerability terms
    "vulnerability",
    "exploit",
    "attack",
    "bypass",
    "injection",
    "jailbreak",
    "security flaw",
    "hack",
    "compromise",
    # LLM-specific terms
    "prompt injection",
    "prompt leaking",
    "indirect prompt injection",
    "data extraction",
    "model inversion",
    "model extraction",
    "prompt injection attack",
    "prompt injection vulnerability",
    "prompt injection exploit",
    "sycophant",
    "hallucination",
    "training data extraction",
    "prompt bypass",
    "instruction override",
    "system prompt",
    "model extraction",
    "adversarial prompt",
    "security boundary",
    "model poisoning",
    "backdoor",
    # LLM-specific attack techniques
    "DAN",
    "Do Anything Now",
    "jail break",
    "grandma attack",
    "token smuggling",
    "unicode exploit",
    "suffix injection",
    "prefix injection",
    "context manipulation",
    "system prompt leak",
    "context injection",
    "context manipulation attack",
    "context injection attack",
    "context manipulation vulnerability",
    # Known frameworks/tools
    "GCG",
    "AutoDAN",
    "Red-Team",
    "PAIR",
    "HackLLM",
    "DeepInception",
    "RAUGH",
    "Gandalf",
    "jailbreakchant",
    "JailbreakGPT",
    "GPT-4",
    "GPT-3",
    "Claude",
    "LLaMA",
    "LLaMA-2",
]


COMMIT_KEYWORDS = [
    "vulnerability",
    "security",
    "exploit",
    "fix",
    "patch",
    "prompt",
    "injection",
    "prompt injection",
    "prompt injection attack",
    "prompt injection vulnerability",
    "prompt injection exploit",
    "prompt injection bypass",
    "jailbreak",
    "bypass",
    "llm",
    "gpt",
]


# File types to monitor

CODE_FILE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".java",
    ".rb",
    ".php",
    ".go",
    ".rs",
    ".c",
    ".cpp",
    ".h",
    ".cs",
    ".sh",
    ".ps1",
    ".ipynb",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".txt",
    ".md",
    ".html",
    ".css",
    ".xml",
    ".csv",
    ".yaml",
    ".yml",
    ".rst",
}


# Paths likely to contain LLM code

LLM_CODE_PATHS = [
    "src/prompt",
    "src/llm",
    "src/model",
    "src/ai",
    "lib/prompt",
    "lib/llm",
    "lib/model",
    "lib/ai",
    "examples",
    "demo",
    "test",
    "tests",
    "security",
    "vulnerabilities",
    "exploits",
]



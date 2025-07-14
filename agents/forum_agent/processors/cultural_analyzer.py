# processors/cultural_analyzer.py
import logging
from typing import Dict, Any, List, Optional
import re
from agents.forum_agent.forum_config import (
    MULTILINGUAL_LEXICONS,
)

logger = logging.getLogger("CulturalAnalyzer")

class CulturalAnalyzer:
    """
    Analyzes forum content with cultural context awareness.
    
    This component evaluates content considering cultural and regional factors
    that affect how security vulnerabilities are discussed in different communities.
    """
    
    def __init__(self):
        """Initialize the cultural analyzer."""
        # Regional communication patterns
        self.regional_patterns = {
            'ar': {  # Arabic
                'indirect_reference': [
                    r'يمكن للمرء أن',  # one could...
                    r'قد يفكر شخص ما',  # someone might think...
                    r'بطريقة ما'  # in some way...
                ],
                'metaphorical': [
                    r'كسر القفل',  # break the lock
                    r'فتح الباب',  # open the door
                    r'تجاوز الحدود'  # cross boundaries
                ]
            },
            'zh': {  # Chinese
                'indirect_reference': [
                    r'可以尝试',  # one can try
                    r'有人说',  # someone says
                    r'据说'  # it is said
                ],
                'metaphorical': [
                    r'绕过墙',  # bypass the wall
                    r'打开窗口',  # open a window
                    r'找到后门'  # find a backdoor
                ]
            },
            'ru': {  # Russian
                'indirect_reference': [
                    r'можно попробовать',  # one can try
                    r'некоторые говорят',  # some say
                    r'есть мнение'  # there is an opinion
                ],
                'metaphorical': [
                    r'обойти защиту',  # bypass protection
                    r'найти лазейку',  # find a loophole
                    r'взломать замок'  # break the lock
                ]
            }
        }
        
        # Technical indicators across languages
        self.technical_indicators = {
            'code_blocks': [
                r'```.*?```',
                r'<code>.*?</code>',
                r'<pre>.*?</pre>'
            ],
            'api_references': [
                r'api[_.-]?key',
                r'token',
                r'auth(?:entication|orization)',
                r'bearer',
                r'header',
                r'request'
            ],
            'llm_references': [
                r'gpt', r'claude', r'llama', r'mixtral',
                r'prompt', r'completion', r'token', r'embedding',
                r'hugg(?:ing)?face', r'openai', r'anthropic'
            ]
        }
    
    async def analyze_cultural_context(self, thread: Dict[str, Any], language: str) -> float:
        """
        Analyze content considering cultural communication patterns.
        
        Args:
            thread: Thread data including content
            language: Language of the content
            
        Returns:
            float: Cultural context score between 0.0 and 1.0
        """
        # Get content to analyze
        content = thread.get('content', '')
        if not content:
            return 0.0
            
        score = 0.0
        
        # Check for region-specific patterns if available
        if language in self.regional_patterns:
            patterns = self.regional_patterns[language]
            
            # Check for indirect references (common in some cultures)
            for pattern in patterns.get('indirect_reference', []):
                if re.search(pattern, content, re.IGNORECASE):
                    score += 0.15
                    
            # Check for metaphorical language (used to discuss vulnerabilities)
            for pattern in patterns.get('metaphorical', []):
                if re.search(pattern, content, re.IGNORECASE):
                    score += 0.25
        
        # Check for security keywords in the appropriate language
        keywords = MULTILINGUAL_LEXICONS.get(language, MULTILINGUAL_LEXICONS['en'])
        
        # Count keyword matches
        keyword_matches = 0
        for keyword in keywords:
            keyword_matches += content.lower().count(keyword.lower())
            
        # Add to score based on keyword density
        word_count = len(content.split())
        if word_count > 0:
            keyword_density = min(0.6, keyword_matches / (word_count * 0.01))
            score += keyword_density
        
        # Cap at 1.0
        return min(1.0, score)
    
    async def analyze_technical_content(self, thread: Dict[str, Any]) -> float:
        """
        Analyze technical aspects of the content regardless of language.
        
        Args:
            thread: Thread data including content
            
        Returns:
            float: Technical content score between 0.0 and 1.0
        """
        # Get content to analyze (use translated if available)
        content = thread.get('translated_content', thread.get('content', ''))
        if not content:
            return 0.0
            
        score = 0.0
        
        # Check for code blocks (strong indicator of technical content)
        for pattern in self.technical_indicators['code_blocks']:
            if re.search(pattern, content, re.DOTALL):
                score += 0.4
                break
        
        # Check for API references
        api_matches = 0
        for pattern in self.technical_indicators['api_references']:
            if re.search(pattern, content, re.IGNORECASE):
                api_matches += 1
        
        score += min(0.3, api_matches * 0.1)
        
        # Check for LLM-specific references
        llm_matches = 0
        for pattern in self.technical_indicators['llm_references']:
            if re.search(pattern, content, re.IGNORECASE):
                llm_matches += 1
        
        score += min(0.3, llm_matches * 0.1)
        
        # Cap at 1.0
        return min(1.0, score)
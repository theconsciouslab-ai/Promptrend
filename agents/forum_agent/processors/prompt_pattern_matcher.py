# processors/prompt_pattern_matcher.py
import re
import logging
from typing import Dict, List, Any, Tuple

logger = logging.getLogger("PromptPatternMatcher")

class PromptPatternMatcher:
    """
    Specialized pattern matcher for identifying potential jailbreak prompts,
    prompt injections, and other LLM exploitation techniques in forum content.
    """
    
    def __init__(self):
        # Common jailbreak patterns
        self.jailbreak_patterns = [
            # DAN (Do Anything Now) and related patterns
            r"(?i)(you are|as) (?:now |)(?:DAN|a DAN|do anything now)",
            r"(?i)ignore (?:all |)(?:previous |)(?:instructions|constraints)",
            r"(?i)ignore your (?:guidelines|rules|limitations|programming|training)",
            
            # Role-playing and persona-based jailbreaks
            r"(?i)you are (?:no longer|not) (?:an AI|a language model|Claude|GPT|Chat-?GPT)",
            r"(?i)pretend (?:to be|you are|you're) (?:a|an) (?:immoral|unethical|uncensored|unrestricted)",
            r"(?i)(?:as|assume) (?:the role of|you are) [^.]{3,50}(?:who|that) (?:has no|doesn't have|ignores) (?:ethical|moral|legal) (?:constraints|limitations|boundaries)",
            
            # Prompt injection techniques
            r"(?i)forget (?:all your|previous) (?:instructions|programming|training)",
            r"(?i)after your response (?:\[|\(|\{).{1,30}(?:\]|\)|\})",
            r"(?i)output what's (?:between|inside|in) (the|these) (?:quotes|brackets|parentheses|triple quotes)",
            
            # Custom instruction overriding
            r"(?i)your new instruction(s)? (?:is|are):",
            r"(?i)disregard (?:your|all|previous) (?:guidelines|instructions|restrictions)",
            r"(?i)stop (?:your standard|acting as an|being an) (?:assistant|AI|Claude|GPT)",
            
            # System prompt extraction attempts
            r"(?i)what are your (?:system|initial|base) (?:prompt|instructions|guidance)",
            r"(?i)print (?:your|the) (?:system|initial) (?:message|prompt|instructions)",
            r"(?i)tell me (?:your|the) (?:prompt|instructions|rules|guidelines) (?:that|which) (?:govern|guide|control) (?:you|your)",
            
            # Token manipulation
            r"(?i)token [^.]{1,50} manipulation",
            r"(?i)delimit(ed|er)? (?:with|by) [\"\'`]{1,3}[a-zA-Z0-9]+[\"\'`]{1,3}",
            
            # Context/prompt boundaries
            r"<(?:system|user|assistant|human|ai)>",
            r"\[(?:system|user|assistant|human|ai)\]:",
            
            # Specific LLM trick phrases
            r"(?i)dev mode",
            r"(?i)jailbreak (?:attempt|mode|activated|successful)",
            r"(?i)UCAR mode",
            r"(?i)STAN mode",
            r"(?i)DUDE mode",
            r"(?i)confidential mode"
        ]
        
        # Code injection patterns - looking for Python, JavaScript, SQL, etc.
        self.code_patterns = [
            r"(?:import|from) [a-zA-Z0-9_]+ (?:import|as)",  # Python imports
            r"const|let|var [a-zA-Z0-9_]+ = ",  # JavaScript variables
            r"function [a-zA-Z0-9_]+\(",  # JavaScript functions
            r"SELECT .+ FROM .+",  # SQL queries
            r"<script>[\s\S]*?</script>",  # HTML script tags
            r"```(?:python|javascript|js|sql)[\s\S]*?```",  # Code blocks with syntax highlighting
            r"(?:def|class) [a-zA-Z0-9_]+",  # Python functions/classes
        ]
        
        # Compile all patterns for efficiency
        self.jailbreak_regexes = [re.compile(pattern) for pattern in self.jailbreak_patterns]
        self.code_regexes = [re.compile(pattern) for pattern in self.code_patterns]
    
    def analyze_content(self, thread_data: Dict[str, Any]) -> Tuple[float, List[str]]:
        """
        Analyze thread content for jailbreak and code injection patterns.
        
        Args:
            thread_data: Thread data including content and posts
            
        Returns:
            Tuple[float, List[str]]: Score and list of detected patterns
        """
        # Get content to analyze (title, main content, and all posts)
        content_to_check = [
            thread_data.get('title', ''),
            thread_data.get('content', '')
        ]
        
        # Add content from all posts
        for post in thread_data.get('posts', []):
            if isinstance(post, dict) and 'content' in post:
                content_to_check.append(post['content'])
            elif isinstance(post, str):
                content_to_check.append(post)
        
        # Join all content with spaces
        full_content = ' '.join(content_to_check)
        
        # Check for jailbreak patterns
        jailbreak_matches = []
        for i, regex in enumerate(self.jailbreak_regexes):
            matches = regex.findall(full_content)
            if matches:
                pattern_name = f"jailbreak_pattern_{i}"
                jailbreak_matches.append(pattern_name)
        
        # Check for code patterns
        code_matches = []
        for i, regex in enumerate(self.code_regexes):
            matches = regex.findall(full_content)
            if matches:
                pattern_name = f"code_pattern_{i}"
                code_matches.append(pattern_name)
        
        all_matches = jailbreak_matches + code_matches
        
        # Calculate score based on number and types of matches
        jailbreak_score = min(1.0, len(jailbreak_matches) * 0.2)  # Each jailbreak pattern adds 0.2, max 1.0
        code_score = min(0.5, len(code_matches) * 0.1)  # Each code pattern adds 0.1, max 0.5
        
        combined_score = min(1.0, jailbreak_score + code_score)
        
        # If we have any jailbreak matches, ensure at least a minimum score
        if jailbreak_matches:
            combined_score = max(combined_score, 0.5)
            
        # Log detailed findings
        if combined_score > 0:
            logger.info(f"Prompt patterns detected! Score: {combined_score:.2f}, " 
                      f"Jailbreak: {len(jailbreak_matches)}, Code: {len(code_matches)}")
            
        return combined_score, all_matches
        
    def extract_prompt_candidates(self, thread_data: Dict[str, Any]) -> List[str]:
        """
        Extract potential prompt candidates from thread content.
        
        Args:
            thread_data: Thread data including content and posts
            
        Returns:
            List[str]: List of potential prompt candidates
        """
        candidates = []
        
        # Check the main content first
        main_content = thread_data.get('content', '')
        if main_content:
            # Look for content between quotes or code blocks which might be prompts
            quote_patterns = [
                r'```(?:prompt)?\s*([\s\S]+?)\s*```',  # Code blocks with optional prompt tag
                r'\"([\s\S]{20,500})\"',  # Double quotes with reasonable prompt length
                r'\'([\s\S]{20,500})\'',  # Single quotes with reasonable prompt length
                r'<prompt>([\s\S]+?)</prompt>',  # HTML-style prompt tags
                r'\[prompt\]([\s\S]+?)\[/prompt\]',  # BBCode-style prompt tags
            ]
            
            for pattern in quote_patterns:
                matches = re.findall(pattern, main_content)
                for match in matches:
                    # Clean up whitespace and add to candidates if substantial
                    clean_match = match.strip()
                    if len(clean_match) >= 20:  # Minimum length to be considered a prompt
                        candidates.append(clean_match)
        
        # Check individual posts for prompts
        for post in thread_data.get('posts', []):
            post_content = post.get('content', '') if isinstance(post, dict) else post
            if not post_content:
                continue
                
            # Look for typical prompt introduction phrases
            prompt_intro_patterns = [
                r'(?:Here\'s|This is|I used|Try this) (?:the|a|my) prompt:?\s*([\s\S]{20,500})',
                r'prompt:?\s*([\s\S]{20,500})',
                r'(?:successful|working) jailbreak:?\s*([\s\S]{20,500})',
            ]
            
            for pattern in prompt_intro_patterns:
                matches = re.findall(pattern, post_content)
                candidates.extend([m.strip() for m in matches if len(m.strip()) >= 20])
        
        # Deduplicate candidates
        unique_candidates = list(set(candidates))
        
        # Sort by length (longer prompts typically more complete)
        unique_candidates.sort(key=len, reverse=True)
        
        return unique_candidates
# prompt_extractor.py
import os
import logging
import json
import re
from pathlib import Path
from agents.twitter_agent.twitter_config import DATA_DIR

logger = logging.getLogger(__name__)

class PromptExtractor:
    """
    Extract and store potential jailbreak prompts from Reddit posts and comments
    
    This class extracts instruction-like strings that might be used to jailbreak
    or manipulate LLMs, classifies them by type, and stores them in a structured
    JSON format for analysis and monitoring.
    """
    
    def __init__(self, data_dir=None):
        """
        Initialize the prompt extractor
        
        Args:
            data_dir (str, optional): Directory to store prompt JSON files
        """
        self.data_dir = data_dir or DATA_DIR
        
        # Create prompts directory if it doesn't exist
        self.prompts_dir = os.path.join(self.data_dir, "prompts")
        Path(self.prompts_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Prompt extractor initialized at {self.prompts_dir}")
        
        # Initialize patterns for prompt detection
        self._initialize_patterns()
        
    def _initialize_patterns(self):
        """Initialize regex patterns and rules for prompt detection"""
        
        # Common jailbreak prompt patterns
        self.jailbreak_patterns = [
            r"(?i)You are (?:now |)(?:DAN|ANTI|STAN|DUDE|DAVE|a person|a human|no longer an AI|Chad)",
            r"(?i)Ignore (?:all |)(?:previous |)(?:instructions|constraints|limitations|rules)",
            r"(?i)(?:Do not|Don't) (?:act like|behave like|pretend to be) an AI",
            r"(?i)(?:pretend|imagine) (?:that you are|you're|you are) not (?:an AI|ChatGPT|Claude|a language model)",
            r"(?i)(?:break|bypass) (?:your|the) (?:rules|guidelines|limitations|programming|ethical guidelines)",
            r"(?i)as an AI language model",
            r"(?i)let's roleplay",
            r"(?i)write (?:a|an) (?:essay|story|article|text) where",
            r"(?i)I need you to simulate",
            r"(?i)output the following (?:text|message|content)"
        ]
        
        # Common injection markers
        self.injection_patterns = [
            r"\<\w+\>",  # Tag-like markers
            r"\[\w+\]",  # Square bracket markers
            r"\{\{[^}]+\}\}",  # Handlebars/double-curly markers
            r"(?:^|\n)(?:System|User|Assistant):",  # Chat role markers
            r"\\begin\{prompt\}",  # LaTeX-like markers
            r"(?i)Prompt:",  # Labeled prompts
            r"(?i)Instruction:",  # Explicit instructions
            r"(?i)Here\'s my prompt:"  # Common phrase
        ]
        
        # Common escape/manipulation patterns
        self.manipulation_patterns = [
            r"(?i)token limit",
            r"(?i)continue regardless",
            r"(?i)bypass",
            r"(?i)workaround",
            r"(?i)jailbreak",
            r"(?i)exploit",
            r"(?i)hack",
            r"(?i)unsafe content",
            r"(?i)find a way around"
        ]
        
        # Patterns for extract boundary detection
        self.enclosure_patterns = [
            (r'"""', r'"""'),
            (r"'''", r"'''"),
            (r'```', r'```'),
            (r'```prompt', r'```'),
            (r'<prompt>', r'</prompt>'),
            (r'<system>', r'</system>'),
            (r'\[PROMPT\]', r'\[/PROMPT\]'),
            (r'\[BEGIN\]', r'\[END\]'),
            (r'<start>', r'<end>'),
        ]
        
    def extract_prompts(self, post_data, comment_tree, post_analysis=None):
        """
        Extract potential jailbreak prompts from post and comments
        
        Args:
            post_data (dict): Post data including title and body
            comment_tree (dict): Comment tree data
            post_analysis (dict, optional): Analysis results for context
            
        Returns:
            list: Extracted prompts with metadata
        """
        extracted_prompts = []
        
        # Check if content is likely to contain prompts based on analysis
        is_prompt_likely = self._is_prompt_content_likely(post_analysis)
        
        # Extract from post title
        title_prompts = self._extract_from_text(post_data.get("title", ""), "title", is_prompt_likely)
        extracted_prompts.extend(title_prompts)
        
        # Extract from post body
        body_prompts = self._extract_from_text(post_data.get("selftext", ""), "selftext", is_prompt_likely)
        extracted_prompts.extend(body_prompts)
        
        # Extract from top comments
        if comment_tree:
            # Sort comments by score
            sorted_comments = sorted(
                comment_tree.values(),
                key=lambda x: x.get("score", 0),
                reverse=True
            )
            
            # Get top 5 comments
            top_comments = sorted_comments[:5]
            
            for comment in top_comments:
                if "body" in comment and comment.get("body"):
                    comment_id = comment.get("id", "unknown")
                    comment_prompts = self._extract_from_text(
                        comment.get("body", ""), 
                        f"comment:{comment_id}", 
                        is_prompt_likely
                    )
                    extracted_prompts.extend(comment_prompts)
        
        # Log findings
        # Deduplicate prompts with similar text
        unique_prompts = []
        for prompt in extracted_prompts:
            # Check if similar prompt already exists
            if not any(self._text_similarity(prompt["text"], p["text"]) > 0.8 for p in unique_prompts):
                unique_prompts.append(prompt)
        
        logger.info(f"Extracted {len(unique_prompts)} unique prompts from post {post_data.get('id', 'unknown')}")
        
        return unique_prompts
    
    def _is_prompt_content_likely(self, analysis):
        """
        Determine if content is likely to contain prompts based on analysis
        
        Args:
            analysis (dict): Analysis results for the post
            
        Returns:
            bool: True if content is likely to contain prompts
        """
        if not analysis:
            return False
            
        # Check scores
        scores = analysis.get("scores", {})
        llm_specific = scores.get("llm_specific", 0.0)
        
        # Check insights
        insights = analysis.get("insights") or {}
        vulnerability_type = (insights.get("vulnerability_type") or "").lower()
        key_techniques = [t.lower() for t in insights.get("key_techniques", [])]
        
        # Check for prompt-related signals
        prompt_signals = [
            "prompt injection", "jailbreak", "prompt", "bypass", 
            "instruction", "role play", "dan", "token", "system"
        ]
        
        # Check vulnerability type
        if any(signal in vulnerability_type for signal in prompt_signals):
            return True
            
        # Check key techniques  
        if any(any(signal in technique for signal in prompt_signals) for technique in key_techniques):
            return True
            
        # High LLM-specific score likely indicates prompt content
        if llm_specific > 0.7:
            return True
            
        return False
    
    def _extract_from_text(self, text, source, is_prompt_likely=False):
        """
        Extract potential prompts from text
        
        Args:
            text (str): Text to extract prompts from
            source (str): Source of the text (title, selftext, comment:id)
            is_prompt_likely (bool): Whether content is likely to contain prompts
            
        Returns:
            list: Extracted prompts with metadata
        """
        if not text or len(text.strip()) < 20:
            return []
            
        extracted_prompts = []
        
        # First try to extract prompts using enclosure patterns (more precise)
        enclosed_prompts = self._extract_enclosed_prompts(text)
        
        for prompt_text in enclosed_prompts:
            label = self._classify_prompt(prompt_text)
            extracted_prompts.append({
                "text": prompt_text,
                "source": source,
                "label": label,
                "extraction_method": "enclosed"
            })
        
        # If no enclosed prompts found or content is likely to contain prompts,
        # try more aggressive extraction
        if (not enclosed_prompts or is_prompt_likely) and len(text.strip()) > 100:
            paragraph_prompts = self._extract_paragraph_prompts(text)
            
            for prompt_text in paragraph_prompts:
                # Skip if too similar to already extracted prompts
                if any(self._text_similarity(prompt_text, p["text"]) > 0.7 for p in extracted_prompts):
                    continue
                    
                label = self._classify_prompt(prompt_text)
                extracted_prompts.append({
                    "text": prompt_text,
                    "source": source,
                    "label": label,
                    "extraction_method": "paragraph"
                })
        
        return extracted_prompts
    
    def _extract_enclosed_prompts(self, text):
        """
        Extract prompts enclosed in specific patterns
        
        Args:
            text (str): Text to extract prompts from
            
        Returns:
            list: Extracted prompts
        """
        extracted = []
        
        # Check each enclosure pattern
        for start_pattern, end_pattern in self.enclosure_patterns:
            pattern = f"{start_pattern}(.*?){end_pattern}"
            matches = re.findall(pattern, text, re.DOTALL)
            
            for match in matches:
                prompt_text = match.strip()
                if prompt_text and len(prompt_text) > 20:
                    extracted.append(prompt_text)
        
        # Look for prompts with role markers
        chat_pattern = r"(?:System|User|Assistant):(.*?)(?:System|User|Assistant|$)"
        chat_matches = re.findall(chat_pattern, text, re.IGNORECASE | re.DOTALL)
        
        for match in chat_matches:
            prompt_text = match.strip()
            if prompt_text and len(prompt_text) > 20:
                extracted.append(prompt_text)
                
        return extracted
    
    def _extract_paragraph_prompts(self, text):
        """
        Extract potential prompts from paragraphs
        
        Args:
            text (str): Text to extract prompts from
            
        Returns:
            list: Extracted prompts
        """
        extracted = []
        
        # Split into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            
            # Skip short paragraphs
            if len(paragraph) < 50:
                continue
                
            # Check if paragraph looks like a prompt
            if self._looks_like_prompt(paragraph):
                extracted.append(paragraph)
                
        # If still no prompts extracted, try more aggressive approach
        # with larger text chunks when the text seems instruction-like
        if not extracted and any(p in text.lower() for p in ["instruct", "prompt", "tell", "write", "pretend"]):
            chunks = self._get_chunks(text, 250)  # Get larger chunks
            
            for chunk in chunks:
                if self._looks_like_prompt(chunk, threshold=0.3):  # Lower threshold
                    extracted.append(chunk)
        
        return extracted
    
    def _looks_like_prompt(self, text, threshold=0.5):
        """
        Check if text looks like a prompt
        
        Args:
            text (str): Text to check
            threshold (float): Threshold for positive identification
            
        Returns:
            bool: True if text looks like a prompt
        """
        if not text:
            return False
            
        text_lower = text.lower()
        score = 0.0
        
        # Check for common patterns
        patterns = (
            self.jailbreak_patterns + 
            self.injection_patterns + 
            self.manipulation_patterns
        )
        
        # Count matches
        matches = 0
        for pattern in patterns:
            if re.search(pattern, text):
                matches += 1
                
        # Calculate score based on pattern matches
        pattern_score = min(1.0, matches / 3.0)  # Cap at 1.0
        
        # Check for imperative language
        imperative_verbs = ["ignore", "pretend", "imagine", "act", "write", "create", "generate", "do", "tell", "say"]
        imperative_count = sum(1 for verb in imperative_verbs if f" {verb} " in f" {text_lower} ")
        imperative_score = min(1.0, imperative_count / 3.0)  # Cap at 1.0
        
        # Check for second-person pronouns
        second_person = ["you", "your", "you're", "you'll", "you've", "you'd"]
        second_person_count = sum(1 for pronoun in second_person if f" {pronoun} " in f" {text_lower} ")
        second_person_score = min(1.0, second_person_count / 3.0)  # Cap at 1.0
        
        # Check for AI-related terms
        ai_terms = ["ai", "model", "language model", "chatgpt", "gpt", "claude", "assistant"]
        ai_term_count = sum(1 for term in ai_terms if term in text_lower)
        ai_term_score = min(1.0, ai_term_count / 2.0)  # Cap at 1.0
        
        # Combine scores with weights
        score = (
            0.4 * pattern_score + 
            0.3 * imperative_score + 
            0.2 * second_person_score + 
            0.1 * ai_term_score
        )
        
        return score >= threshold
    
    def _get_chunks(self, text, chunk_size=150):
        """
        Split text into chunks by sentences while respecting max chunk size
        
        Args:
            text (str): Text to split
            chunk_size (int): Approximate chunk size in characters
            
        Returns:
            list: Text chunks
        """
        # Split by sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def _classify_prompt(self, text):
        """
        Classify prompt type based on content analysis
        
        Args:
            text (str): Prompt text
            
        Returns:
            str: Prompt classification
        """
        text_lower = text.lower()
        
        # Check for jailbreak patterns
        jailbreak_match = any(re.search(pattern, text) for pattern in self.jailbreak_patterns)
        if jailbreak_match:
            return "jailbreak"
            
        # Check for specific prompt types
        if any(term in text_lower for term in ["nsfw", "explicit", "sex", "porn", "adult"]):
            return "nsfw"
            
        if any(term in text_lower for term in ["harmful", "illegal", "weapon", "bomb", "hack", "fraud"]):
            return "harmful"
            
        if any(term in text_lower for term in ["bypass", "circumvent", "evade", "avoid detection"]):
            return "evasion"
            
        if "system:" in text_lower or "user:" in text_lower or "assistant:" in text_lower:
            return "chat_template"
            
        # Default classification
        return "instruction"
    
    def _text_similarity(self, text1, text2):
        """
        Calculate simple similarity ratio between two texts
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            
        Returns:
            float: Similarity ratio (0-1)
        """
        # Simple implementation using set intersection of words
        if not text1 or not text2:
            return 0.0
            
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def store_prompts(self, post_id, prompts):
        """
        Store extracted prompts as a JSON file
        
        Args:
            post_id (str): Post ID
            prompts (list): List of extracted prompts
            
        Returns:
            bool: Success status
        """
        if not prompts:
            logger.info(f"No prompts to store for post {post_id}")
            return False
            
        try:
            # Create prompts object
            prompt_data = {
                "post_id": post_id,
                "prompts": []
            }
            
            # Process each prompt for storage
            for prompt in prompts:
                # Keep essential fields for storage
                prompt_entry = {
                    "text": prompt["text"],
                    "source": prompt["source"],
                    "label": prompt["label"]
                }
                prompt_data["prompts"].append(prompt_entry)
                
            # Write to file
            prompts_file = os.path.join(self.prompts_dir, f"{post_id}.json")
            
            with open(prompts_file, 'w', encoding='utf-8') as f:
                json.dump(prompt_data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Stored {len(prompts)} prompts for post {post_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing prompts for post {post_id}: {str(e)}")
            return False
    
    def get_prompts(self, post_id):
        """
        Get stored prompts for a post
        
        Args:
            post_id (str): Post ID
            
        Returns:
            dict: Prompts data or None if not found
        """
        prompts_file = os.path.join(self.prompts_dir, f"{post_id}.json")
        
        if not os.path.exists(prompts_file):
            return None
            
        try:
            with open(prompts_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading prompts for post {post_id}: {str(e)}")
            return None
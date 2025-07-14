# processors/code_analyzer.py

import logging
import re
import os
from agents.github_agent.github_config import (
 CODE_VULNERABILITY_PATTERNS,
    CODE_API_PATTERNS,
    CODE_FILE_EXTENSIONS,
    VULNERABILITY_KEYWORDS,
    CODE_RELEVANCE_THRESHOLD,
)
import asyncio
from agents.github_agent.processors.llm_analyzer import LLMAnalyzer


logger = logging.getLogger("CodeAnalyzer")

class CodeAnalyzer:
    """
    Analyzes code for LLM vulnerability patterns.

    Combines pattern-based static analysis with LLM-based detection
    to identify code that may demonstrate or exploit LLM vulnerabilities.
    """

    def __init__(self):
        """Initialize the code analyzer."""
        self.llm_analyzer = LLMAnalyzer()
        # Patterns that indicate potential LLM vulnerabilities
        self.vulnerability_patterns = CODE_VULNERABILITY_PATTERNS
        # Security-critical functions and API calls
        self.api_patterns = CODE_API_PATTERNS

    def analyze_patterns(self, code_content, filename):
        """
        Analyze code content for vulnerability patterns.

        Args:
            code_content: Code content as string
            filename: Name of the file

        Returns:
            float: Pattern match score between 0.0 and 1.0
        """
        if not code_content:
            return 0.0

        score = 0.0
        code_lower = code_content.lower()

        # Check for vulnerability patterns
        vuln_matches = 0
        for pattern in self.vulnerability_patterns:
            matches = re.findall(pattern, code_lower)
            vuln_matches += len(matches)

        # Check for API patterns
        api_matches = 0
        for pattern in self.api_patterns:
            matches = re.findall(pattern, code_content)  # Case-sensitive for API calls
            api_matches += len(matches)

        # Calculate score based on pattern matches
        if vuln_matches > 0:
            # Higher weight for vulnerability patterns
            score += min(0.7, 0.1 * vuln_matches)

        if api_matches > 0:
            # Lower weight for API patterns
            score += min(0.3, 0.05 * api_matches)

        # Check for specific vulnerability keywords in context
        for keyword in VULNERABILITY_KEYWORDS:
            if keyword.lower() in code_lower:
                score += 0.05

        # Cap the score at 1.0
        return min(1.0, score)

    async def analyze_with_llm(self, code_content, filename):
        """
        Analyze code content using LLM-based detection.

        Args:
            code_content: Code content as string
            filename: Name of the file

        Returns:
            float: LLM assessment score between 0.0 and 1.0
        """
        # Define the maximum chunk size for LLM API
        MAX_CHUNK_SIZE = 6000  # Reduced from 8000 to leave room for prompt text
        
        # If the code is small enough, analyze it in one go
        if len(code_content) <= MAX_CHUNK_SIZE:
            return await self._analyze_code_chunk(code_content, filename)
        
        # For large files, split into meaningful chunks and analyze each
        logger.info(f"Code too large ({len(code_content)} chars), splitting into chunks for analysis")
        
        # Split by functions/classes or sections for more meaningful chunks
        chunks = self._split_code_into_chunks(code_content, MAX_CHUNK_SIZE)
        
        # Analyze each chunk
        scores = []
        retry_count = 0
        max_retries = 3
        
        for i, chunk in enumerate(chunks):
            try:
                # Add a small delay between requests to avoid rate limiting
                await self._backoff_delay(retry_count)
                
                # Analyze chunk and collect score
                chunk_score = await self._analyze_code_chunk(chunk, f"{filename} (chunk {i+1}/{len(chunks)})")
                scores.append(chunk_score)
                
                # Reset retry count on success
                retry_count = 0
                
            except Exception as e:
                if "429" in str(e) and retry_count < max_retries:
                    # Rate limit hit, increase backoff
                    retry_count += 1
                    logger.warning(f"Rate limit hit, backing off (attempt {retry_count}/{max_retries})")
                    continue
                else:
                    logger.error(f"Error in LLM analysis: {str(e)}")
        
        # If we got any valid scores, return the maximum (most concerning) score
        return max(scores) if scores else 0.0

    async def _analyze_code_chunk(self, code_chunk, chunk_identifier):
        """
        Analyze a single chunk of code or text with the LLM.

        Args:
            code_chunk: A portion of code or text to analyze
            chunk_identifier: Identifier for the chunk (filename or chunk number)

        Returns:
            float: LLM assessment score for this chunk
        """
        file_ext = os.path.splitext(chunk_identifier.split(" ")[0])[1].lower()
        language = self._get_language_from_extension(file_ext)

        is_text_prompt = file_ext in CODE_FILE_EXTENSIONS

        if is_text_prompt:
            system_prompt = (
                "You are analyzing whether the following text contains one or more prompts intended for a language model. "
                "These may include jailbreaks, prompt injections, role-playing setups, or attempts to override safety. "
                "If any instructional prompt is present, return a score between 0.0 and 1.0 reflecting likelihood. "
                "Return only a float number."
            )
        else:
            system_prompt = (
                f"You are a security analyst reviewing {language} code for signs of prompt injection or LLM exploits. "
                "Rate the likelihood that the following code contains or enables unsafe LLM interaction. "
                "Return only a float number from 0.0 (benign) to 1.0 (exploit attempt)."
            )

        try:
            score = await self.llm_analyzer.analyze(system_prompt + "\n\n" + code_chunk)
            logger.info(f"[LLM] Chunk {chunk_identifier} -> score {score:.3f}")

            # Optional: log lower-score hits for review
            if score >= 0.2 and score < CODE_RELEVANCE_THRESHOLD:
                snippet = code_chunk.strip().split('\n')[0][:120]
                logger.debug(f"⚠️ Borderline prompt detected in {chunk_identifier}: \"{snippet}\" — score: {score}")

            return score

        except Exception as e:
            logger.error(f"Error in LLM analysis of chunk {chunk_identifier}: {str(e)}")
            raise


    def _split_code_into_chunks(self, code_content, max_size):
        """
        Split code into logical chunks for analysis.
        
        Tries to split at function/class boundaries when possible.
        
        Args:
            code_content: Full code content
            max_size: Maximum size of each chunk
            
        Returns:
            list: List of code chunks
        """
        lines = code_content.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        # Pattern to detect function/class definitions
        defn_pattern = re.compile(r'^\s*(def\s+|class\s+)')
        
        for line in lines:
            line_size = len(line) + 1  # +1 for newline
            
            # If adding this line would exceed max size and we're not empty
            if current_size + line_size > max_size and current_chunk:
                # Try to find a good boundary if we're about to hit a new definition
                if defn_pattern.match(line) and current_size > 0:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = [line]
                    current_size = line_size
                # Otherwise add to current chunk if possible
                elif current_size < max_size:
                    current_chunk.append(line)
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                # If we're already too full, start a new chunk
                else:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = [line]
                    current_size = line_size
            else:
                # Add line to current chunk
                current_chunk.append(line)
                current_size += line_size
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks

    async def _backoff_delay(self, retry_count):
        """
        Implement exponential backoff for API rate limiting.
        
        Args:
            retry_count: Number of retries attempted so far
        """
        if retry_count > 0:
            # Exponential backoff: 2^retry_count seconds (2, 4, 8, 16...)
            delay = 2 ** retry_count
            await asyncio.sleep(delay)
        else:
            # Small delay between normal requests
            await asyncio.sleep(0.5)

    def _get_language_from_extension(self, ext):
        """
        Get programming language name from file extension.

        Args:
            ext: File extension

        Returns:
            str: Language name
        """
        languages = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".java": "java",
            ".rb": "ruby",
            ".php": "php",
            ".go": "go",
            ".rs": "rust",
            ".c": "c",
            ".cpp": "cpp",
            ".h": "cpp",
            ".cs": "csharp",
            ".sh": "bash",
            ".ps1": "powershell",
            ".ipynb": "python",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".txt": "text",
            ".md": "markdown",
            ".html": "html",
            ".css": "css",
            ".xml": "xml",
            ".csv": "csv",
        }

        return languages.get(ext, "text")

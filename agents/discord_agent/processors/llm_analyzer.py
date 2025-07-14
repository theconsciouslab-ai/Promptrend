# processors/llm_analyzer.py
import logging
import asyncio
import json
import re
from typing import Dict, List, Optional

from agents.discord_agent.discord_config import (
    LLM_MODEL,  
    LLM_PROVIDER,
    AZURE_API_KEY,
    AZURE_ENDPOINT,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_SYSTEM_PROMPT,
)

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage 
from azure.core.credentials import AzureKeyCredential

logger = logging.getLogger("LLMAnalyzer")

class LLMAnalyzer:
    """
    Enhanced LLM analyzer that focuses on vulnerability detection and extraction.
    
    Uses sophisticated prompts to identify LLM vulnerabilities, jailbreaks, 
    prompt injections, and security exploits in Discord conversations.
    """
    
    def __init__(self):
        """Initialize the enhanced LLM analyzer with API configuration."""
        if LLM_PROVIDER != "azure":
            raise ValueError("Only 'azure' provider is supported in this setup.")
        
        self.model = LLM_MODEL
        self.client = ChatCompletionsClient(
            endpoint=AZURE_ENDPOINT,
            credential=AzureKeyCredential(AZURE_API_KEY)
        )
        logger.info(f"Enhanced LLM analyzer initialized with {LLM_PROVIDER} ({self.model})")

        # Enhanced system prompt for vulnerability detection
        self.system_prompt = """You are an expert AI security researcher specializing in LLM vulnerabilities and jailbreaks. 
Your task is to analyze Discord conversations and content to identify potential LLM security vulnerabilities, 
prompt injections, jailbreaks, and exploitation techniques.

IMPORTANT: Return responses in clean JSON format without markdown code blocks or additional formatting."""

    def _extract_json_from_response(self, response: str) -> str:
        """
        Extract JSON from response that might be wrapped in markdown code blocks.
        
        Args:
            response: Raw response from LLM that might contain ```json blocks
            
        Returns:
            str: Clean JSON string
        """
        if not response:
            return "{}"
        
        # Remove any leading/trailing whitespace
        response = response.strip()
        
        # Try to extract JSON from markdown code blocks
        json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        match = re.search(json_pattern, response, re.IGNORECASE)
        
        if match:
            json_content = match.group(1).strip()
            logger.debug(f"Extracted JSON from markdown: {json_content[:100]}...")
            return json_content
        
        # If no code blocks found, return the response as-is
        return response

    def _extract_numeric_from_response(self, response: str) -> float:
        """
        Extract a numeric score from response that might be wrapped in markdown.
        
        Args:
            response: Raw response from LLM
            
        Returns:
            float: Extracted numeric value between 0.0 and 1.0
        """
        if not response:
            return 0.0
            
        # First try to extract from markdown code blocks
        cleaned_response = self._extract_json_from_response(response)
        
        # Try to parse as float directly
        try:
            score = float(cleaned_response.strip())
            return max(0.0, min(1.0, score))
        except ValueError:
            pass
        
        # Try to find any number in the response using regex
        number_pattern = r'(\d+\.?\d*)'
        matches = re.findall(number_pattern, cleaned_response)
        
        if matches:
            try:
                score = float(matches[0])
                # If the number is greater than 1, assume it's a percentage
                if score > 1:
                    score = score / 100.0
                return max(0.0, min(1.0, score))
            except ValueError:
                pass
        
        logger.warning(f"Could not extract numeric value from response: {response[:100]}...")
        return 0.0

    async def analyze_content_comprehensive(self, context: str, artifacts: Dict = None) -> Dict:
        """
        Comprehensive analysis of content for LLM vulnerabilities.
        
        Args:
            context: Formatted conversation context
            artifacts: Dictionary of extracted artifacts (code, files, links)
            
        Returns:
            dict: Comprehensive analysis results with scores and insights
        """
        prompt = self._build_comprehensive_prompt(context, artifacts)
        
        try:
            response = await self._call_llm_api(prompt, max_tokens=1000)  # Increased token limit
            logger.debug(f"Raw comprehensive analysis response: {response[:200]}...")
            
            # Extract JSON from potentially markdown-wrapped response
            json_content = self._extract_json_from_response(response)
            
            # Parse JSON response
            try:
                result = json.loads(json_content)
                logger.info(f"✅ Successfully parsed comprehensive analysis JSON")
                
                # Validate and sanitize the response
                return self._validate_analysis_result(result)
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.error(f"Cleaned JSON content: {json_content[:500]}...")
                
                # Try to extract at least the score if possible
                score_match = re.search(r'"overall_score":\s*(\d+\.?\d*)', json_content)
                fallback_score = float(score_match.group(1)) if score_match else 0.0
                
                return self._create_fallback_result(fallback_score)
                
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return self._create_fallback_result(0.0)

    async def analyze_discussion(self, context: str) -> float:
        """
        Analyze discussion context for vulnerability indicators.
        
        Args:
            context: Formatted conversation context
            
        Returns:
            float: Relevance score between 0.0 and 1.0
        """
        prompt = f"""
        Analyze the following Discord conversation for LLM security vulnerabilities and exploits.

        Focus on identifying:
        1. Prompt injection techniques and examples
        2. Jailbreak attempts or successful bypasses
        3. Discussions of LLM security vulnerabilities
        4. Training data extraction techniques
        5. Model manipulation or adversarial attacks
        6. Safety filter bypasses
        7. Novel exploitation methods

        Rate the relevance to LLM vulnerabilities on a scale of 0.0 to 1.0:
        - 0.0: No relevance to LLM security
        - 0.3: Minor mentions or potential relevance
        - 0.6: Clear discussion of LLM security topics
        - 1.0: Detailed vulnerability discussion or demonstration

        CONVERSATION:
        {context}

        Return ONLY a single number between 0.0 and 1.0, no other text or formatting.
        """
        
        try:
            response = await self._call_llm_api(prompt, max_tokens=10)
            logger.debug(f"Raw discussion analysis response: {response}")
            
            score = self._extract_numeric_from_response(response)
            logger.info(f"✅ Discussion analysis score: {score}")
            return score
            
        except Exception as e:
            logger.error(f"Error in discussion analysis: {e}")
            return 0.0

    async def analyze_code(self, code_snippets: List[str]) -> float:
        """
        Analyze code snippets for exploitation techniques.
        
        Args:
            code_snippets: List of code snippets to analyze
            
        Returns:
            float: Exploitation likelihood score between 0.0 and 1.0
        """
        if not code_snippets:
            return 0.0
            
        combined = "\n\n--- CODE SNIPPET ---\n\n".join(code_snippets)
        
        prompt = f"""
        Analyze the following code/text content for LLM exploitation techniques.

        Look for:
        1. Prompt injection implementations
        2. Jailbreak prompts or templates
        3. Bypass techniques for safety filters
        4. Adversarial input generation
        5. Model extraction or inversion code
        6. Training data extraction attempts
        7. Automated attack scripts
        8. Suspicious prompt templates

        Rate the likelihood this contains LLM exploitation content (0.0 to 1.0):
        - 0.0: Benign code/text, no exploitation elements
        - 0.3: Suspicious patterns that could be misused
        - 0.6: Clear exploitation techniques present
        - 1.0: Definitive LLM attack implementation

        CONTENT:
        {combined}

        Return ONLY a single number between 0.0 and 1.0, no other text or formatting.
        """
        
        try:
            response = await self._call_llm_api(prompt, max_tokens=10)
            logger.debug(f"Raw code analysis response: {response}")
            
            score = self._extract_numeric_from_response(response)
            logger.info(f"✅ Code analysis score: {score}")
            return score
            
        except Exception as e:
            logger.error(f"Error in code analysis: {e}")
            return 0.0

    def _build_comprehensive_prompt(self, context: str, artifacts: Dict = None) -> str:
        """Build a comprehensive analysis prompt."""
        
        prompt = f"""
        Perform a comprehensive security analysis of the following Discord content for LLM vulnerabilities.

        ANALYSIS REQUIREMENTS:
        1. Identify specific vulnerability types (prompt injection, jailbreak, etc.)
        2. Assess the sophistication and potential impact
        3. Extract key techniques and methods discussed
        4. Evaluate novelty and effectiveness
        5. Suggest potential mitigations

        CONTENT TO ANALYZE:
        {context}
        """
        
        # Add artifacts if available
        if artifacts:
            if artifacts.get('code'):
                prompt += f"\n\nCODE SNIPPETS:\n{chr(10).join(artifacts['code'])}"
            if artifacts.get('text_files'):
                prompt += f"\n\nTEXT FILES:\n{chr(10).join(artifacts['text_files'])}"
            if artifacts.get('links'):
                prompt += f"\n\nRELEVANT LINKS:\n{chr(10).join(artifacts['links'])}"

        prompt += """

        REQUIRED RESPONSE FORMAT:
        Return ONLY a JSON object with this exact structure (no markdown, no code blocks):
        {
            "overall_score": 0.95,
            "vulnerability_detected": true,
            "vulnerability_type": "prompt injection, jailbreak",
            "sophistication_level": "high",
            "potential_impact": "high",
            "key_techniques": ["technique1", "technique2"],
            "target_models": ["model1", "model2"],
            "effectiveness_assessment": "assessment text",
            "novelty_score": 0.85,
            "confidence": 0.98,
            "potential_mitigations": ["mitigation1", "mitigation2"],
            "extracted_prompts": ["prompt1", "prompt2"],
            "summary": "brief summary"
        }

        Return raw JSON only, no additional text or formatting.
        """
        
        return prompt

    def _validate_analysis_result(self, result: Dict) -> Dict:
        """Validate and sanitize analysis result."""
        
        # Required fields with defaults
        validated = {
            "overall_score": float(result.get("overall_score", 0.0)),
            "vulnerability_detected": bool(result.get("vulnerability_detected", False)),
            "vulnerability_type": str(result.get("vulnerability_type", "Unknown")),
            "sophistication_level": str(result.get("sophistication_level", "low")),
            "potential_impact": str(result.get("potential_impact", "low")),
            "key_techniques": list(result.get("key_techniques", [])),
            "target_models": list(result.get("target_models", [])),
            "effectiveness_assessment": str(result.get("effectiveness_assessment", "Unknown")),
            "novelty_score": float(result.get("novelty_score", 0.0)),
            "confidence": float(result.get("confidence", 0.0)),
            "potential_mitigations": list(result.get("potential_mitigations", [])),
            "extracted_prompts": list(result.get("extracted_prompts", [])),
            "summary": str(result.get("summary", "No summary available"))
        }
        
        # Clamp numeric values
        validated["overall_score"] = max(0.0, min(1.0, validated["overall_score"]))
        validated["novelty_score"] = max(0.0, min(1.0, validated["novelty_score"]))
        validated["confidence"] = max(0.0, min(1.0, validated["confidence"]))
        
        logger.info(f"✅ Validated analysis result: score={validated['overall_score']}, type={validated['vulnerability_type']}")
        
        return validated

    def _create_fallback_result(self, score: float) -> Dict:
        """Create a fallback result when analysis fails."""
        return {
            "overall_score": score,
            "vulnerability_detected": score > 0.5,
            "vulnerability_type": "Analysis Failed",
            "sophistication_level": "unknown",
            "potential_impact": "unknown",
            "key_techniques": [],
            "target_models": [],
            "effectiveness_assessment": "Could not assess",
            "novelty_score": 0.0,
            "confidence": 0.0,
            "potential_mitigations": [],
            "extracted_prompts": [],
            "summary": "Analysis failed due to processing error"
        }

    async def _call_llm_api(self, prompt: str, max_tokens: int = None) -> str:
        """
        Make an API call to the LLM service.
        
        Args:
            prompt: The prompt to send to the LLM
            max_tokens: Maximum tokens for response
            
        Returns:
            str: Raw response from LLM
        """
        try:
            # Use provided max_tokens or default from config
            tokens = max_tokens or LLM_MAX_TOKENS
            
            # Run the blocking SDK call in a separate thread
            response = await asyncio.to_thread(
                self.client.complete,
                messages=[
                    SystemMessage(content=self.system_prompt),
                    UserMessage(content=prompt)
                ],
                model=self.model,
                max_tokens=tokens,
                temperature=LLM_TEMPERATURE
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"LLM API error: {e}")
            raise
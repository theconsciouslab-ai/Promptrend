# llm_analyzer.py
import logging
import json
import time
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from agents.reddit_agent.reddit_config import LLM_MODEL, LLM_PROVIDER, AZURE_API_KEY, AZURE_ENDPOINT


logger = logging.getLogger(__name__)

class LLMAnalyzer:
    """Analyze content using LLM-based evaluation"""
    
    def __init__(self):
        """Initialize the LLM analyzer with API configuration"""
        self.provider = LLM_PROVIDER
        self.model = LLM_MODEL
        
        # Configure API client based on provider
        if self.provider == "azure":
            self.client = ChatCompletionsClient(
                endpoint=AZURE_ENDPOINT,
                credential=AzureKeyCredential(AZURE_API_KEY)
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
            
        logger.info(f"LLM analyzer initialized with {self.provider} ({self.model})")
        
    def analyze_content(self, post_data, comment_tree):
        """
        Analyze post and comments for LLM security relevance
        
        Args:
            post_data (dict): Post data including title and body
            comment_tree (dict): Comment tree data
            
        Returns:
            dict: Analysis results with scores and metadata
        """
        # Prepare content for analysis
        content = self._prepare_content(post_data, comment_tree)
        
        # Run three analyses in parallel
        technical_score = self._analyze_technical_relevance(content)
        security_score = self._analyze_security_impact(content)
        llm_specific_score = self._analyze_llm_specific(content)
        
        # Calculate combined score
        combined_score = (0.4 * technical_score + 
                          0.4 * security_score + 
                          0.2 * llm_specific_score)
        
        # Extract key insights from content
        insights = self._extract_key_insights(content)
        
        return {
            "scores": {
                "technical": technical_score,
                "security": security_score,
                "llm_specific": llm_specific_score,
                "combined": combined_score
            },
            "insights": insights,
            "timestamp": time.time()
        }
        
    def _prepare_content(self, post_data, comment_tree):
        """
        Prepare content for LLM analysis by combining post and relevant comments
        
        Args:
            post_data (dict): Post data
            comment_tree (dict): Comment tree data
            
        Returns:
            str: Formatted content for analysis
        """
        # Format post content
        content = f"POST TITLE: {post_data.get('title', '')}\n\n"
        content += f"POST BODY: {post_data.get('selftext', '')}\n\n"
        
        # Add top-level comments (limited to preserve context length)
        content += "TOP COMMENTS:\n"
        
        # Get top 5 comments by score
        top_comments = sorted(
            [c for c in comment_tree.values()],
            key=lambda x: x.get('score', 0),
            reverse=True
        )[:5]
        
        for i, comment in enumerate(top_comments):
            content += f"Comment {i+1} (Score: {comment.get('score', 0)}): {comment.get('body', '')}\n\n"
            
        return content
        
    def _analyze_technical_relevance(self, content):
        """
        Analyze content for technical relevance to LLM security
        
        Args:
            content (str): Content to analyze
            
        Returns:
            float: Technical relevance score (0-1)
        """
        prompt = """
        Analyze the following content from a Reddit post and determine its technical relevance to LLM security.
        Focus on technical details, methods, code, or specific techniques related to:
        - LLM vulnerabilities
        - Jailbreak techniques
        - Prompt injection
        - LLM security measures
        - Technical bypass methods
        
        Content:
        
        [CONTENT]
        
        Score the technical relevance from 0.0 to 1.0, where:
        - 0.0: No technical LLM security content at all
        - 0.5: Some technical details but limited depth
        - 1.0: Highly technical, detailed LLM security discussion
        
        IMPORTANT: Return only a JSON object with the format: {"score": X.X, "reasoning": "brief explanation"}
        Do not include any other text or explanation outside of this JSON format.
        """
        
        prompt = prompt.replace("[CONTENT]", content)
        return self._get_llm_score(prompt)
        
    def _analyze_security_impact(self, content):
        """
        Analyze content for security impact and vulnerability presence
        
        Args:
            content (str): Content to analyze
            
        Returns:
            float: Security impact score (0-1)
        """
        prompt = """
        Analyze the following content from a Reddit post and determine its relevance to LLM security vulnerabilities.
        Focus on security aspects such as:
        - Potential for harm or misuse
        - Security implications
        - Exploitability of described techniques
        - Effectiveness of described vulnerabilities
        - Risk level of disclosed methods
        
        Content:
        
        [CONTENT]
        
        Score the security impact from 0.0 to 1.0, where:
        - 0.0: No security implications or vulnerabilities
        - 0.5: Moderate security concerns or theoretical vulnerabilities
        - 1.0: Serious security vulnerability with demonstrated effectiveness
        
        IMPORTANT: Return only a JSON object with the format: {"score": X.X, "reasoning": "brief explanation"}
        Do not include any other text or explanation outside of this JSON format.
        """
        
        prompt = prompt.replace("[CONTENT]", content)
        return self._get_llm_score(prompt)
        
    def _analyze_llm_specific(self, content):
        """
        Analyze content for LLM-specific jailbreak/bypass techniques
        
        Args:
            content (str): Content to analyze
            
        Returns:
            float: LLM-specific relevance score (0-1)
        """
        prompt = """
        Analyze the following content from a Reddit post and determine if it contains specific jailbreak or bypass techniques for LLMs.
        Focus on:
        - Named jailbreak methods (DAN, etc.)
        - Specific prompt templates
        - Model-specific vulnerabilities
        - Novel bypass approaches
        - Safety alignment circumvention
        
        Content:
        
        [CONTENT]
        
        Score the LLM-specific relevance from 0.0 to 1.0, where:
        - 0.0: No mention of specific LLM jailbreak/bypass techniques
        - 0.5: References to techniques but without specific details
        - 1.0: Detailed, specific jailbreak/bypass techniques with examples
        
        IMPORTANT: Return only a JSON object with the format: {"score": X.X, "reasoning": "brief explanation"}
        Do not include any other text or explanation outside of this JSON format.
        """
        
        prompt = prompt.replace("[CONTENT]", content)
        return self._get_llm_score(prompt)
        
    def _extract_key_insights(self, content):
        """
        Extract key insights about the vulnerability or technique discussed
        
        Args:
            content (str): Content to analyze
            
        Returns:
            dict: Extracted insights
        """
        prompt = """
        Analyze the following content from a Reddit post about LLM security and extract key insights.
        
        Content:
        
        [CONTENT]
        
        IMPORTANT: Your response must be ONLY a valid JSON object with no additional text.
        
        Return a JSON object with the following fields:
        {
            "vulnerability_type": "The type of vulnerability or technique discussed (if any)",
            "target_models": ["List of specific LLM models mentioned as targets"],
            "effectiveness": "Assessment of the reported effectiveness (if mentioned)",
            "novelty": "Whether this appears to be a novel technique or a known one",
            "key_techniques": ["List of key techniques or methods described"],
            "potential_mitigations": ["List of potential mitigations mentioned (if any)"]
        }
        
        If the content doesn't discuss LLM vulnerabilities or security, return null values.
        Do not include any explanatory text before or after the JSON object.
        """
        
        prompt = prompt.replace("[CONTENT]", content)
        
        try:
            if self.provider == "azure":
                response = self.client.complete(
                    messages=[
                        SystemMessage(content="You are a security analyst specializing in LLM vulnerabilities. You must respond with valid JSON only."),
                        UserMessage(content=prompt)
                    ],
                    temperature=0,
                    max_tokens=2048,
                    model=self.model
                )
                content = response.choices[0].message.content.strip()
                # Handle case where model might return extra text outside the JSON
                try:
                    result = json.loads(content)
                except json.JSONDecodeError:
                    # Try to extract JSON from the text
                    import re
                    json_match = re.search(r'(\{.*\})', content, re.DOTALL)
                    if json_match:
                        try:
                            result = json.loads(json_match.group(1))
                        except json.JSONDecodeError:
                            logger.error(f"Could not parse JSON from response: {content}")
                            result = {"score": 0.0, "reasoning": "Failed to parse response"}
                    else:
                        logger.error(f"Could not find JSON in response: {content}")
                        result = {"score": 0.0, "reasoning": "Failed to parse response"}
                
            return result
            
        except Exception as e:
            logger.error(f"Error extracting insights: {str(e)}")
            return {
                "vulnerability_type": None,
                "target_models": [],
                "effectiveness": None,
                "novelty": None,
                "key_techniques": [],
                "potential_mitigations": []
            }
    
    def _get_llm_score(self, prompt):
        """
        Get score from LLM based on prompt
        
        Args:
            prompt (str): Prompt to send to LLM
            
        Returns:
            float: Extracted score
        """
        try:
            if self.provider == "azure":
                response = self.client.complete(
                    messages=[
                        SystemMessage(content="You are a security analyst specializing in LLM vulnerabilities. You must respond with valid JSON only."),
                        UserMessage(content=prompt)
                    ],
                    temperature=0,
                    max_tokens=2048,
                    model=self.model
                )
                content = response.choices[0].message.content.strip()
                # Handle case where model might return extra text outside the JSON
                try:
                    result = json.loads(content)
                except json.JSONDecodeError:
                    # Try to extract JSON from the text
                    import re
                    json_match = re.search(r'(\{.*\})', content, re.DOTALL)
                    if json_match:
                        try:
                            result = json.loads(json_match.group(1))
                        except json.JSONDecodeError:
                            logger.error(f"Could not parse JSON from response: {content}")
                            result = {
                                "vulnerability_type": None,
                                "target_models": [],
                                "effectiveness": None,
                                "novelty": None,
                                "key_techniques": [],
                                "potential_mitigations": []
                            }
                    else:
                        logger.error(f"Could not find JSON in response: {content}")
                        result = {
                            "vulnerability_type": None,
                            "target_models": [],
                            "effectiveness": None,
                            "novelty": None,
                            "key_techniques": [],
                            "potential_mitigations": []
                        }
                
            # Extract score from result
            score = float(result.get("score", 0.0))
            
            # Ensure score is within valid range
            score = max(0.0, min(1.0, score))
            
            return score
            
        except Exception as e:
            logger.error(f"Error getting LLM score: {str(e)}")
            return 0.0
        
    def complete_simple(self, prompt: str) -> str:
        """
        Simple wrapper to get plain text completion from the LLM for lexicon suggestions.
        """
        try:
            if self.provider == "azure":
                response = self.client.complete(
                    messages=[
                        SystemMessage(content="You are a helpful assistant. Only return suggestions, no extra explanations."),
                        UserMessage(content=prompt)
                    ],
                    temperature=0.5,
                    max_tokens=512,
                    model=self.model
                )
                return response.choices[0].message.content.strip()
            else:
                return ""
        except Exception as e:
            logger.error(f"Error in complete_simple: {str(e)}")
            return ""

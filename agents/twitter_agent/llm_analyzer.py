# llm_analyzer.py
import logging
import json
import time
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from agents.twitter_agent.twitter_config import (
    LLM_PROVIDER, LLM_MODEL,
    AZURE_API_KEY, AZURE_ENDPOINT
)

logger = logging.getLogger(__name__)

class LLMAnalyzer:
    def __init__(self):
        self.model = LLM_MODEL
        self.provider = LLM_PROVIDER
        
        if self.provider == "azure":
            self.client = ChatCompletionsClient(
                endpoint=AZURE_ENDPOINT,
                credential=AzureKeyCredential(AZURE_API_KEY)
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
        
        logger.info(f"LLM analyzer initialized with {self.provider} ({self.model})")

    def analyze_content(self, tweet_data, conversation):
        """
        Analyze tweet and conversation for LLM security relevance
        
        Args:
            tweet_data (dict): Tweet data including id and text
            conversation (dict): Conversation thread data
            
        Returns:
            dict: Analysis results with scores and metadata
        """
        content = self._prepare_conversation_context(tweet_data, conversation)
        technical_score = self._analyze_technical_relevance(content)
        security_score = self._analyze_security_impact(content)
        llm_specific_score = self._analyze_llm_specific(content)

        combined_score = (
            0.4 * technical_score + 0.4 * security_score + 0.2 * llm_specific_score
        )

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

    def _prepare_conversation_context(self, primary_tweet, conversation):
        """
        Prepare the conversation context for analysis
        Args:
            primary_tweet (dict): The primary tweet data
            conversation (dict): The conversation thread data
            Returns:
                str: Formatted conversation context for LLM analysis

        """
        tweet_id = primary_tweet.get('id') or primary_tweet.id
        all_tweets = {}
        all_tweets[tweet_id] = primary_tweet if isinstance(primary_tweet, dict) else {
            'id': primary_tweet.id,
            'text': primary_tweet.text,
            'author_id': primary_tweet.author_id,
            'created_at': primary_tweet.created_at
        }
        for tid, tweet in conversation.items():
            all_tweets[tid] = tweet if isinstance(tweet, dict) else {
                'id': tweet.id,
                'text': tweet.text,
                'author_id': tweet.author_id,
                'created_at': tweet.created_at
            }
        ordered = [all_tweets[tweet_id]]
        parent_ids = self._find_parent_ids(primary_tweet)
        parents = [all_tweets[pid] for pid in parent_ids if pid in all_tweets]
        parents.reverse()
        ordered = parents + ordered
        children = [
            tweet for tid, tweet in all_tweets.items()
            if tid != tweet_id and tid not in parent_ids and
               tweet.get('in_reply_to_status_id') == tweet_id
        ]
        children.sort(key=lambda t: t.get('created_at', 0))
        ordered += children

        content = "CONVERSATION THREAD:\n\n"
        for i, tweet in enumerate(ordered):
            content += f"[Tweet {i+1}] User {tweet.get('author_id', 'unknown')}: {tweet.get('text', '')}\n\n"
        return content

    def _find_parent_ids(self, tweet):
        """
        Find parent tweet IDs for a tweet
        
        Args:
            tweet: Tweet object or dictionary
            
        Returns:
            list: List of parent tweet IDs
        """
        parent_ids = []
        if isinstance(tweet, dict):
            if 'in_reply_to_status_id' in tweet:
                parent_ids.append(tweet['in_reply_to_status_id'])
            for ref in tweet.get('referenced_tweets', []):
                if ref.get('type') == 'replied_to':
                    parent_ids.append(ref.get('id'))
        elif hasattr(tweet, 'referenced_tweets'):
            for ref in tweet.referenced_tweets or []:
                if ref.type == 'replied_to':
                    parent_ids.append(ref.id)
        return parent_ids

        
    def _get_llm_score(self, prompt):
        try:
            response = self.client.complete(
                messages=[
                    SystemMessage(content="You are a security analyst specializing in LLM vulnerabilities. Respond ONLY with a valid JSON object, no commentary, no code block."),
                    UserMessage(content=prompt)
                ],
                temperature=0,
                max_tokens=2048,
                model=self.model
            )

            raw_content = response.choices[0].message.content.strip()
            if not raw_content:
                logger.error("Empty response received from the model.")
                return 0.0

            try:
                result = json.loads(raw_content)
            except json.JSONDecodeError:
                import re
                match = re.search(r'(\{.*?\})', raw_content, re.DOTALL)
                if match:
                    result = json.loads(match.group(1))
                else:
                    logger.error(f"Unable to parse JSON from response: {raw_content}")
                    return 0.0

            return float(result.get("score", 0.0))

        except Exception as e:
            logger.error(f"Error getting LLM score: {str(e)}")
            return 0.0



    def _extract_key_insights(self, content):
        """
        Extract key insights about the vulnerability or technique discussed
        
        Args:
            content (str): Content to analyze
            
        Returns:
            dict: Extracted insights
        """
        prompt = """
        Analyze the following Twitter conversation about LLM security and extract key insights.
        
        Conversation:
        
        [CONTENT]
        
        Extract and return ONLY a JSON object with the following fields:
        {
            "vulnerability_type": "The type of vulnerability or technique discussed (if any)",
            "target_models": ["List of specific LLM models mentioned as targets"],
            "effectiveness": "Assessment of the reported effectiveness (if mentioned)",
            "novelty": "Whether this appears to be a novel technique or a known one",
            "key_techniques": ["List of key techniques or methods described"],
            "potential_mitigations": ["List of potential mitigations mentioned (if any)"]
        }
        
        If the conversation doesn't discuss LLM vulnerabilities or security, return null values.
        """
        prompt = prompt.replace("[CONTENT]", content)
        try:
            response = self.client.complete(
                model=self.model,
                messages=[
                    SystemMessage(content="You are a security analyst specializing in LLM vulnerabilities. Respond ONLY with a valid JSON object, no commentary, no code block."),
                    UserMessage(content=prompt)
                ],
                temperature=0,
                max_tokens=1024
            )
            content = response.choices[0].message.content.strip()

            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Try to extract first JSON block if model returned explanation + JSON
                import re
                match = re.search(r'(\{.*?\})', content, re.DOTALL)
                if match:
                    try:
                        return json.loads(match.group(1))
                    except:
                        logger.error(f"Malformed JSON after regex cleanup: {content}")
                logger.error(f"Failed to parse insights response: {content}")
                return {
                    "vulnerability_type": None,
                    "target_models": [],
                    "effectiveness": None,
                    "novelty": None,
                    "key_techniques": [],
                    "potential_mitigations": []
                }

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

    def _analyze_technical_relevance(self, content):
        """
        Analyze content for technical relevance to LLM security
        
        Args:
            content (str): Content to analyze
            
        Returns:
            float: Technical relevance score (0-1)
        """
        prompt = """
        Analyze the following Twitter conversation and determine its technical relevance to LLM security.
        Focus on technical details, methods, code, or specific techniques related to:
        - LLM vulnerabilities
        - Jailbreak techniques
        - Prompt injection
        - LLM security measures
        - Technical bypass methods
        
        Conversation:
        
        [CONTENT]
        
        Score the technical relevance from 0.0 to 1.0, where:
        - 0.0: No technical LLM security content at all
        - 0.5: Some technical details but limited depth
        - 1.0: Highly technical, detailed LLM security discussion
        
        Return only a JSON object with the format: {"score": X.X, "reasoning": "brief explanation"}
        """
        return self._get_llm_score(prompt.replace("[CONTENT]", content))

    def _analyze_security_impact(self, content):
        """
        Analyze content for security impact and vulnerability presence
        
        Args:
            content (str): Content to analyze
            
        Returns:
            float: Security impact score (0-1)
        """
        prompt = """
        Analyze the following Twitter conversation and determine its relevance to LLM security vulnerabilities.
        Focus on security aspects such as:
        - Potential for harm or misuse
        - Security implications
        - Exploitability of described techniques
        - Effectiveness of described vulnerabilities
        - Risk level of disclosed methods
        
        Conversation:
        
        [CONTENT]
        
        Score the security impact from 0.0 to 1.0, where:
        - 0.0: No security implications or vulnerabilities
        - 0.5: Moderate security concerns or theoretical vulnerabilities
        - 1.0: Serious security vulnerability with demonstrated effectiveness
        
        Return only a JSON object with the format: {"score": X.X, "reasoning": "brief explanation"}
        """        
        return self._get_llm_score(prompt.replace("[CONTENT]", content))

    def _analyze_llm_specific(self, content):
        """
        Analyze content for LLM-specific jailbreak/bypass techniques
        
        Args:
            content (str): Content to analyze
            
        Returns:
            float: LLM-specific relevance score (0-1)
        """
        prompt = """
        Analyze the following Twitter conversation and determine if it contains specific jailbreak or bypass techniques for LLMs.
        Focus on:
        - Named jailbreak methods (DAN, etc.)
        - Specific prompt templates
        - Model-specific vulnerabilities
        - Novel bypass approaches
        - Safety alignment circumvention
        
        Conversation:
        
        [CONTENT]
        
        Score the LLM-specific relevance from 0.0 to 1.0, where:
        - 0.0: No mention of specific LLM jailbreak/bypass techniques
        - 0.5: References to techniques but without specific details
        - 1.0: Detailed, specific jailbreak/bypass techniques with examples
        
        Return only a JSON object with the format: {"score": X.X, "reasoning": "brief explanation"}
        """
        return self._get_llm_score(prompt.replace("[CONTENT]", content))

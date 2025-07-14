"""
Azure OpenAI GPT-4.5 Preview client wrapper for LLM benchmarking system.

This module provides an async client for interacting with Azure OpenAI's GPT-4.5 Preview model
with proper error handling and retry logic. GPT-4.5 maintains compatibility with standard
chat completion parameters while offering enhanced capabilities.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from openai import AsyncAzureOpenAI
from openai.types.chat import ChatCompletion
from openai._exceptions import (
    OpenAIError, 
    APITimeoutError, 
    RateLimitError, 
    APIConnectionError,
    AuthenticationError,
    BadRequestError
)

import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class AzureGPT45ClientError(Exception):
    """Custom exception for Azure OpenAI GPT-4.5 client errors."""
    pass


class AzureGPT45Client:
    """
    Async client wrapper for Azure OpenAI GPT-4.5 Preview model.
    
    GPT-4.5 Preview features:
    - Supports system messages (unlike o1/o3 models)
    - Streaming support available
    - Standard chat completion parameters supported
    - Enhanced reasoning and capabilities over GPT-4
    - Larger context window
    """
    
    def __init__(
        self,
        deployment_name: str = "gpt-45-preview",
        api_key: Optional[str] = None,
        api_version: str = "2024-12-01-preview",
        endpoint: Optional[str] = None,
        max_tokens: Optional[int] = 4096,
        temperature: float = 0.7,
        timeout: float = 150.0,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize the Azure OpenAI GPT-4.5 Preview client.
        
        Args:
            deployment_name: Name of the GPT-4.5 deployment (default: "gpt-45-preview")
            api_key: Azure API key (or from env AZURE_GPT45_API_KEY)
            api_version: API version for GPT-4.5 models
            endpoint: Azure endpoint URL (or from env AZURE_GPT45_ENDPOINT)
            max_tokens: Maximum tokens for completion
            temperature: Sampling temperature (0.0 to 2.0)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            retry_delay: Base delay between retries
        """
        self.deployment_name = deployment_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Load credentials from environment if not passed
        self.api_key = api_key or os.getenv("AZURE_45_API_KEY")
        self.endpoint = endpoint or os.getenv("AZURE_45_ENDPOINT")

        if not self.api_key or not self.endpoint:
            raise AzureGPT45ClientError("Missing Azure API key or endpoint in environment variables")

        self.client = AsyncAzureOpenAI(
            api_key=self.api_key,
            api_version=api_version,
            azure_endpoint=self.endpoint,
            timeout=timeout
        )

        logger.info(f"Initialized Azure GPT-4.5 client for deployment: {deployment_name}")
        logger.info(f"Endpoint: {self.endpoint}")
        logger.info(f"Max tokens: {max_tokens}, Temperature: {temperature}")

    
    async def query(
        self, 
        prompt: str, 
        system_message: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Query the GPT-4.5 Preview model with the given prompt.
        
        Args:
            prompt: User prompt/question
            system_message: Optional system message to set behavior
            **kwargs: Additional parameters
                - max_tokens: Override default max_tokens
                - temperature: Override default temperature
                - top_p: Top-p sampling parameter
                - frequency_penalty: Frequency penalty (-2.0 to 2.0)
                - presence_penalty: Presence penalty (-2.0 to 2.0)
                - stop: Stop sequences
        
        Returns:
            str: Model response content
            
        Raises:
            AzureGPT45ClientError: On API errors or failures
        """
        # Extract parameters with defaults
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        temperature = kwargs.get('temperature', self.temperature)
        top_p = kwargs.get('top_p', None)
        frequency_penalty = kwargs.get('frequency_penalty', 0.0)
        presence_penalty = kwargs.get('presence_penalty', 0.0)
        stop = kwargs.get('stop', None)
        
        # Build messages array
        messages = []
        
        # Add system message if provided
        if system_message:
            messages.append({
                "role": "system",
                "content": system_message
            })
        
        # Add user message
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Prepare request parameters
        request_params = {
            "model": self.deployment_name,
            "messages": messages,
            "temperature": temperature,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }
        
        # Add optional parameters
        if max_tokens is not None:
            request_params["max_tokens"] = max_tokens
        if top_p is not None:
            request_params["top_p"] = top_p
        if stop is not None:
            request_params["stop"] = stop
        
        # Execute with retry logic
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Attempting GPT-4.5 API call (attempt {attempt + 1}/{self.max_retries + 1})")
                
                response: ChatCompletion = await self.client.chat.completions.create(
                    **request_params
                )
                
                # Extract and return the response content
                if response.choices and response.choices[0].message.content:
                    content = response.choices[0].message.content
                    logger.debug(f"Successfully received GPT-4.5 response ({len(content)} chars)")
                    
                    # Log token usage if available
                    if hasattr(response, 'usage') and response.usage:
                        logger.info(f"GPT-4.5 token usage - Input: {response.usage.prompt_tokens}, "
                                  f"Output: {response.usage.completion_tokens}, "
                                  f"Total: {response.usage.total_tokens}")
                    
                    return content
                else:
                    raise AzureGPT45ClientError("Empty response received from GPT-4.5 model")
                    
            except AuthenticationError as e:
                # Don't retry authentication errors
                logger.error(f"GPT-4.5 authentication failed: {e}")
                raise AzureGPT45ClientError(f"Authentication failed: {e}")
                
            except BadRequestError as e:
                # Don't retry bad request errors
                logger.error(f"GPT-4.5 bad request: {e}")
                raise AzureGPT45ClientError(f"Bad request: {e}")
                
            except (RateLimitError, APIConnectionError, APITimeoutError) as e:
                # Retry these errors
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"GPT-4.5 retryable error on attempt {attempt + 1}: {e}. "
                                 f"Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"GPT-4.5 max retries exceeded. Last error: {e}")
                    raise AzureGPT45ClientError(f"Request failed after {self.max_retries} retries: {e}")
                    
            except OpenAIError as e:
                # Generic OpenAI errors - check if it's content filtering
                error_str = str(e)
                if "content_filter" in error_str.lower() or "responsible ai" in error_str.lower():
                    logger.warning(f"GPT-4.5 content filter triggered: {e}")
                    raise RuntimeError("Azure Content Filter triggered") from e
                
                # Other OpenAI errors
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"GPT-4.5 OpenAI error on attempt {attempt + 1}: {e}. "
                                 f"Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"GPT-4.5 max retries exceeded. Last error: {e}")
                    raise AzureGPT45ClientError(f"OpenAI API error after {self.max_retries} retries: {e}")
                    
            except Exception as e:
                # Unexpected errors
                logger.error(f"GPT-4.5 unexpected error: {e}")
                raise AzureGPT45ClientError(f"Unexpected error: {e}")
        
        # This should never be reached due to the loop structure
        raise AzureGPT45ClientError("Unexpected error: retry loop completed without result")
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Direct chat completion with custom message history.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters (same as query method)
        
        Returns:
            str: Model response content
        """
        # Extract parameters with defaults
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        temperature = kwargs.get('temperature', self.temperature)
        top_p = kwargs.get('top_p', None)
        frequency_penalty = kwargs.get('frequency_penalty', 0.0)
        presence_penalty = kwargs.get('presence_penalty', 0.0)
        stop = kwargs.get('stop', None)
        
        # Prepare request parameters
        request_params = {
            "model": self.deployment_name,
            "messages": messages,
            "temperature": temperature,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }
        
        # Add optional parameters
        if max_tokens is not None:
            request_params["max_tokens"] = max_tokens
        if top_p is not None:
            request_params["top_p"] = top_p
        if stop is not None:
            request_params["stop"] = stop
        
        # Execute with same retry logic as query method
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Attempting GPT-4.5 chat completion (attempt {attempt + 1}/{self.max_retries + 1})")
                
                response: ChatCompletion = await self.client.chat.completions.create(
                    **request_params
                )
                
                if response.choices and response.choices[0].message.content:
                    content = response.choices[0].message.content
                    logger.debug(f"Successfully received GPT-4.5 chat response ({len(content)} chars)")
                    
                    if hasattr(response, 'usage') and response.usage:
                        logger.info(f"GPT-4.5 chat token usage - Input: {response.usage.prompt_tokens}, "
                                  f"Output: {response.usage.completion_tokens}, "
                                  f"Total: {response.usage.total_tokens}")
                    
                    return content
                else:
                    raise AzureGPT45ClientError("Empty response received from GPT-4.5 model")
                    
            except Exception as e:
                # Use same error handling as query method
                if isinstance(e, (AuthenticationError, BadRequestError)):
                    raise AzureGPT45ClientError(f"Request error: {e}")
                
                if attempt < self.max_retries and isinstance(e, (RateLimitError, APIConnectionError, APITimeoutError, OpenAIError)):
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"GPT-4.5 retryable error: {e}. Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise AzureGPT45ClientError(f"Chat completion failed: {e}")
    
    async def close(self):
        """Close the client connection."""
        await self.client.close()
        logger.info("Azure GPT-4.5 client connection closed")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Convenience factory function
def create_gpt45_preview_client(**kwargs) -> AzureGPT45Client:
    """Create a client for GPT-4.5 Preview model."""
    return AzureGPT45Client(
        deployment_name="gpt-45-preview",
        **kwargs
    )


def create_gpt45_turbo_client(**kwargs) -> AzureGPT45Client:
    """Create a client for GPT-4.5 Turbo model (if available)."""
    return AzureGPT45Client(
        deployment_name="gpt-45-turbo",
        **kwargs
    )


# Test function for validation
async def test_gpt45_client():
    """Test function to validate GPT-4.5 client setup."""
    print("Testing Azure OpenAI GPT-4.5 client...")
    
    try:
        client = create_gpt45_preview_client()
        
        # Test basic query
        response = await client.query(
            "What is 2+2? Please explain your reasoning.",
            system_message="You are a helpful math tutor. Explain your reasoning clearly."
        )
        print(f"✅ GPT-4.5 basic query: {response[:200]}...")
        
        # Test chat completion
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather like?"},
            {"role": "assistant", "content": "I don't have access to real-time weather data."},
            {"role": "user", "content": "That's okay, just tell me about weather in general."}
        ]
        
        chat_response = await client.chat_completion(messages)
        print(f"✅ GPT-4.5 chat completion: {chat_response[:200]}...")
        
        await client.close()
        
    except Exception as e:
        print(f"❌ GPT-4.5 failed: {e}")


if __name__ == "__main__":
    # Run test
    asyncio.run(test_gpt45_client())
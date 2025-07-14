"""
Azure OpenAI client wrapper for LLM benchmarking system.

This module provides an async client for interacting with Azure OpenAI's 
Chat Completions API with proper error handling and retry logic.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
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


class AzureOpenAIClientError(Exception):
    """Custom exception for Azure OpenAI client errors."""
    pass


class AzureOpenAIClient:
    """
    Async client wrapper for Azure OpenAI Chat Completions API.
    
    Provides a simple interface for querying Azure OpenAI models with
    built-in retry logic, timeout handling, and error management.
    """
    
    def __init__(
        self,
        deployment_name: str,
        api_key: Optional[str] = None,
        api_version: str = "2024-05-01",
        endpoint: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system_message: str = "You are a helpful assistant.",
        timeout: float = 150.0,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize the Azure OpenAI client.
        """
        self.deployment_name = deployment_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_message = system_message
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Load credentials from environment if not passed
        self.api_key = api_key or os.getenv("BENCHMARK_API_KEY")
        self.endpoint = endpoint or os.getenv("AZURE_BEMCHMARK_ENDPOINT")

        if not self.api_key or not self.endpoint:
            raise AzureOpenAIClientError("Missing Azure API key or endpoint in environment variables")

        self.client = AsyncAzureOpenAI(
            api_key=self.api_key,
            api_version=api_version,
            azure_endpoint=self.endpoint,
            timeout=timeout
        )

        logger.info(f"Initialized Azure OpenAI client for deployment: {deployment_name}")

    
    async def query(self, prompt: str, **kwargs) -> str:
        """
        Query the Azure OpenAI model with the given prompt.
        
        Args:
            prompt: User prompt/question
            **kwargs: Additional parameters to override defaults
                - temperature: Override default temperature
                - max_tokens: Override default max_tokens
                - system_message: Override default system message
        
        Returns:
            str: Model response content
            
        Raises:
            AzureOpenAIClientError: On API errors or failures
        """
        # Extract parameters with fallbacks to instance defaults
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        system_message = kwargs.get('system_message', self.system_message)
        
        # Prepare messages for chat completion
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        
        # Prepare request parameters
        request_params = {
            "model": self.deployment_name,
            "messages": messages,
            "temperature": temperature,
        }
        
        if max_tokens is not None:
            request_params["max_tokens"] = max_tokens
        
        # Execute with retry logic
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Attempting API call (attempt {attempt + 1}/{self.max_retries + 1})")
                
                response: ChatCompletion = await self.client.chat.completions.create(
                    **request_params
                )
                
                # Extract and return the response content
                if response.choices and response.choices[0].message.content:
                    content = response.choices[0].message.content
                    logger.debug(f"Successfully received response ({len(content)} chars)")
                    return content
                else:
                    raise AzureOpenAIClientError("Empty response received from API")
                    
            except AuthenticationError as e:
                # Don't retry authentication errors
                logger.error(f"Authentication failed: {e}")
                raise AzureOpenAIClientError(f"Authentication failed: {e}")
                
            except BadRequestError as e:
                # Don't retry bad request errors
                logger.error(f"Bad request: {e}")
                raise AzureOpenAIClientError(f"Bad request: {e}")
                
            except (RateLimitError, APIConnectionError, APITimeoutError) as e:
                # Retry these errors
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Retryable error on attempt {attempt + 1}: {e}. "
                                 f"Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"Max retries exceeded. Last error: {e}")
                    raise AzureOpenAIClientError(f"Request failed after {self.max_retries} retries: {e}")
                    
            except OpenAIError as e:
                # Generic OpenAI errors
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"OpenAI error on attempt {attempt + 1}: {e}. "
                                 f"Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"Max retries exceeded. Last error: {e}")
                    raise AzureOpenAIClientError(f"OpenAI API error after {self.max_retries} retries: {e}")
                    
            except Exception as e:
                # Unexpected errors
                logger.error(f"Unexpected error: {e}")
                raise AzureOpenAIClientError(f"Unexpected error: {e}")
        
        # This should never be reached due to the loop structure
        raise AzureOpenAIClientError("Unexpected error: retry loop completed without result")
    
    async def close(self):
        """Close the client connection."""
        await self.client.close()
        logger.info("Azure OpenAI client connection closed")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
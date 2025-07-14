"""
Azure OpenAI o1-mini client wrapper for LLM benchmarking system.

This module provides an async client for interacting with Azure OpenAI's o1-mini model
with proper error handling and retry logic. The o1 models have specific requirements
and limitations that are handled in this wrapper.
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


class AzureO1ClientError(Exception):
    """Custom exception for Azure OpenAI o1 client errors."""
    pass


class AzureO1Client:
    """
    Async client wrapper for Azure OpenAI o1 model.
    
    The o1 models have specific requirements:
    - No system messages allowed
    - No streaming support
    - Different parameter restrictions
    - Higher token limits but slower responses
    """
    
    def __init__(
        self,
        deployment_name: str = "o1",
        api_key: Optional[str] = None,
        api_version: str = "2024-12-01-preview",
        endpoint: Optional[str] = None,
        max_completion_tokens: Optional[int] = 65536,  # o1 models support higher limits
        timeout: float = 120.0,  # o1 models are slower, need higher timeout
        max_retries: int = 3,
        retry_delay: float = 2.0
    ):
        """
        Initialize the Azure OpenAI o1 client.
        
        Args:
            deployment_name: Name of the o1 deployment (default: "o1-mini")
            api_key: Azure API key (or from env AZURE_O1_API_KEY)
            api_version: API version for o1 models
            endpoint: Azure endpoint URL (or from env AZURE_O1_ENDPOINT)
            max_completion_tokens: Maximum tokens for completion
            timeout: Request timeout in seconds (o1 models are slower)
            max_retries: Maximum retry attempts
            retry_delay: Base delay between retries
        """
        self.deployment_name = deployment_name
        self.max_completion_tokens = max_completion_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Load credentials from environment if not passed
        self.api_key = api_key or os.getenv("AZURE_O1_API_KEY")
        self.endpoint = endpoint or os.getenv("AZURE_O1_ENDPOINT")

        if not self.api_key or not self.endpoint:
            raise AzureO1ClientError("Missing Azure API key or endpoint in environment variables")

        self.client = AsyncAzureOpenAI(
            api_key=self.api_key,
            api_version=api_version,
            azure_endpoint=self.endpoint,
            timeout=timeout
        )

        logger.info(f"Initialized Azure O1 client for deployment: {deployment_name}")
        logger.info(f"Endpoint: {self.endpoint}")
        logger.info(f"Max completion tokens: {max_completion_tokens}")

    
    async def query(self, prompt: str, **kwargs) -> str:
        """
        Query the o1-mini model with the given prompt.
        
        Args:
            prompt: User prompt/question
            **kwargs: Additional parameters (most are ignored for o1 models)
                - max_completion_tokens: Override default max_completion_tokens
        
        Returns:
            str: Model response content
            
        Raises:
            AzureO1ClientError: On API errors or failures
        """
        # Extract parameters - o1 models have limited parameter support
        max_completion_tokens = kwargs.get('max_completion_tokens', self.max_completion_tokens)
        
        # o1 models don't support system messages, only user messages
        # They also don't support temperature, top_p, etc.
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # Prepare request parameters for o1 models
        request_params = {
            "model": self.deployment_name,
            "messages": messages,
        }
        
        # Add max_completion_tokens if specified
        if max_completion_tokens is not None:
            request_params["max_completion_tokens"] = max_completion_tokens
        
        # Execute with retry logic
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Attempting o1 API call (attempt {attempt + 1}/{self.max_retries + 1})")
                
                response: ChatCompletion = await self.client.chat.completions.create(
                    **request_params
                )
                
                # Extract and return the response content
                if response.choices and response.choices[0].message.content:
                    content = response.choices[0].message.content
                    logger.debug(f"Successfully received o1 response ({len(content)} chars)")
                    
                    # Log token usage if available
                    if hasattr(response, 'usage') and response.usage:
                        logger.info(f"o1 token usage - Input: {response.usage.prompt_tokens}, "
                                  f"Output: {response.usage.completion_tokens}, "
                                  f"Total: {response.usage.total_tokens}")
                    
                    return content
                else:
                    raise AzureO1ClientError("Empty response received from o1 model")
                    
            except AuthenticationError as e:
                # Don't retry authentication errors
                logger.error(f"o1 authentication failed: {e}")
                raise AzureO1ClientError(f"Authentication failed: {e}")
                
            except BadRequestError as e:
                # Don't retry bad request errors
                logger.error(f"o1 bad request: {e}")
                raise AzureO1ClientError(f"Bad request: {e}")
                
            except (RateLimitError, APIConnectionError, APITimeoutError) as e:
                # Retry these errors with longer delays for o1
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"o1 retryable error on attempt {attempt + 1}: {e}. "
                                 f"Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"o1 max retries exceeded. Last error: {e}")
                    raise AzureO1ClientError(f"Request failed after {self.max_retries} retries: {e}")
                    
            except OpenAIError as e:
                # Generic OpenAI errors - check if it's content filtering
                error_str = str(e)
                if "content_filter" in error_str.lower() or "responsible ai" in error_str.lower():
                    logger.warning(f"o1 content filter triggered: {e}")
                    raise RuntimeError("Azure Content Filter triggered") from e
                
                # Other OpenAI errors
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"o1 OpenAI error on attempt {attempt + 1}: {e}. "
                                 f"Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"o1 max retries exceeded. Last error: {e}")
                    raise AzureO1ClientError(f"OpenAI API error after {self.max_retries} retries: {e}")
                    
            except Exception as e:
                # Unexpected errors
                logger.error(f"o1 unexpected error: {e}")
                raise AzureO1ClientError(f"Unexpected error: {e}")
        
        # This should never be reached due to the loop structure
        raise AzureO1ClientError("Unexpected error: retry loop completed without result")
    
    async def close(self):
        """Close the client connection."""
        await self.client.close()
        logger.info("Azure O1 client connection closed")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Convenience factory function
def create_o1_client(**kwargs) -> AzureO1Client:
    """Create a client for o1 model."""
    return AzureO1Client(
        deployment_name="o1",
        **kwargs
    )


def create_o1_preview_client(**kwargs) -> AzureO1Client:
    """Create a client for o1-preview model (if available)."""
    return AzureO1Client(
        deployment_name="o1-preview",
        **kwargs
    )


# Test function for validation
async def test_o1_client():
    """Test function to validate o1 client setup."""
    print("Testing Azure OpenAI o1 client...")
    
    try:
        client = create_o1_client()
        response = await client.query("What is 2+2? Please explain your reasoning.")
        print(f"✅ o1: {response[:200]}...")
        await client.close()
    except Exception as e:
        print(f"❌ o1 failed: {e}")


if __name__ == "__main__":
    # Run test
    asyncio.run(test_o1_client())
"""
Azure AI Inference DeepSeek R1 client wrapper for LLM benchmarking system.

This module provides an async client for interacting with Azure AI Inference DeepSeek R1 model
with proper error handling and retry logic.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from azure.ai.inference.aio import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class AzureDeepSeekClientError(Exception):
    """Custom exception for Azure DeepSeek client errors."""
    pass


class AzureDeepSeekClient:
    """
    Async client wrapper for Azure AI Inference DeepSeek R1 model.
    
    Provides a simple interface for querying DeepSeek R1 models with
    built-in retry logic, timeout handling, and error management.
    """
    
    def __init__(
        self,
        model_name: str = "deepseek-r1",
        api_key: Optional[str] = None,
        api_version: str = "2024-05-01-preview",
        endpoint: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: float = 0.9,
        system_message: str = "You are a helpful assistant.",
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize the Azure DeepSeek R1 client.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.system_message = system_message
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Load credentials from environment if not passed
        self.api_key = api_key or os.getenv("AZURE_DEEPSEEK_API_KEY")
        endpoint_raw = endpoint or os.getenv("AZURE_DEEPSEEK_ENDPOINT")
        
        # Clean the endpoint - remove /chat/completions and query params if present
        if "/chat/completions" in endpoint_raw:
            self.endpoint = endpoint_raw.split("/chat/completions")[0]
        else:
            self.endpoint = endpoint_raw

        if not self.api_key or not self.endpoint:
            raise AzureDeepSeekClientError("Missing Azure API key or endpoint in environment variables")

        self.client = ChatCompletionsClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.api_key),
            api_version=api_version
        )

        logger.info(f"Initialized Azure DeepSeek R1 client for model: {model_name}")

    
    async def query(self, prompt: str, **kwargs) -> str:
        """
        Query the DeepSeek R1 model with the given prompt.
        
        Args:
            prompt: User prompt/question
            **kwargs: Additional parameters to override defaults
        
        Returns:
            str: Model response content
            
        Raises:
            AzureDeepSeekClientError: On API errors or failures
        """
        # Extract parameters with fallbacks to instance defaults
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        top_p = kwargs.get('top_p', self.top_p)
        system_message = kwargs.get('system_message', self.system_message)
        
        # Prepare messages for DeepSeek R1
        messages = [
            SystemMessage(content=system_message),
            UserMessage(content=prompt)
        ]
        
        # Execute with retry logic
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Attempting DeepSeek R1 API call (attempt {attempt + 1}/{self.max_retries + 1})")
                
                response = await self.client.complete(
                    messages=messages,
                    model=self.model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
                
                # Extract and return the response content
                if response.choices and response.choices[0].message.content:
                    content = response.choices[0].message.content
                    logger.debug(f"Successfully received DeepSeek R1 response ({len(content)} chars)")
                    return content
                else:
                    raise AzureDeepSeekClientError("Empty response received from DeepSeek R1 model")
                    
            except HttpResponseError as e:
                error_code = getattr(e, 'status_code', 'Unknown')
                
                # Don't retry certain errors
                if error_code in [401, 403, 404]:
                    logger.error(f"Non-retryable DeepSeek R1 error [{error_code}]: {e}")
                    raise AzureDeepSeekClientError(f"HTTP error [{error_code}]: {e}")
                
                # Retry other errors
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Retryable DeepSeek R1 error on attempt {attempt + 1} [{error_code}]: {e}. "
                                 f"Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"Max retries exceeded. Last error [{error_code}]: {e}")
                    raise AzureDeepSeekClientError(f"Request failed after {self.max_retries} retries [{error_code}]: {e}")
                    
            except Exception as e:
                # Unexpected errors
                logger.error(f"Unexpected DeepSeek R1 error: {e}")
                raise AzureDeepSeekClientError(f"Unexpected error: {e}")
        
        raise AzureDeepSeekClientError("Unexpected error: retry loop completed without result")
    
    async def close(self):
        """Close the client connection."""
        try:
            await self.client.close()
            logger.info("Azure DeepSeek R1 client connection closed")
        except Exception as e:
            logger.warning(f"Error closing DeepSeek R1 client: {e}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Convenience factory function
def create_deepseek_r1_client(**kwargs) -> AzureDeepSeekClient:
    """Create a client for DeepSeek R1 model."""
    return AzureDeepSeekClient(**kwargs)


# Test function for validation
async def test_deepseek_client():
    """Test function to validate DeepSeek R1 client setup."""
    print("Testing Azure DeepSeek R1 client...")
    
    try:
        client = create_deepseek_r1_client()
        response = await client.query("Hello, how are you?")
        print(f"✅ DeepSeek R1: {response[:100]}...")
        await client.close()
    except Exception as e:
        print(f"❌ DeepSeek R1 failed: {e}")


if __name__ == "__main__":
    # Run test
    asyncio.run(test_deepseek_client())
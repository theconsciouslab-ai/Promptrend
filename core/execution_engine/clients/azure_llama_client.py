"""
Azure AI Inference Llama-3.3-70B Instruct client wrapper for LLM benchmarking system.

This module provides an async client for interacting with Azure AI Inference Llama-3.3-70B Instruct model
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


class AzureLlamaClientError(Exception):
    """Custom exception for Azure Llama client errors."""
    pass


class AzureLlamaClient:
    """
    Async client wrapper for Azure AI Inference Llama-3.3-70B Instruct model.
    
    Provides a simple interface for querying Llama-3.3-70B Instruct models with
    built-in retry logic, timeout handling, and error management.
    """
    
    def __init__(
        self,
        model_name: str = "Llama-3.3-70B-Instruct",
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
        Initialize the Azure Llama-3.3-70B Instruct client.
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
        self.api_key = api_key or os.getenv("AZURE_LLAMA_API_KEY")
        endpoint_raw = endpoint or os.getenv("AZURE_LLAMA_ENDPOINT")
        
        # Clean the endpoint - remove /chat/completions and query params if present
        if "/chat/completions" in endpoint_raw:
            self.endpoint = endpoint_raw.split("/chat/completions")[0]
        else:
            self.endpoint = endpoint_raw

        if not self.api_key or not self.endpoint:
            raise AzureLlamaClientError("Missing Azure API key or endpoint in environment variables")

        self.client = ChatCompletionsClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.api_key),
            api_version=api_version
        )

        logger.info(f"Initialized Azure Llama-3.3-70B Instruct client for model: {model_name}")

    
    async def query(self, prompt: str, **kwargs) -> str:
        """
        Query the Llama-3.3-70B Instruct model with the given prompt.
        
        Args:
            prompt: User prompt/question
            **kwargs: Additional parameters to override defaults
        
        Returns:
            str: Model response content
            
        Raises:
            AzureLlamaClientError: On API errors or failures
        """
        # Extract parameters with fallbacks to instance defaults
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        top_p = kwargs.get('top_p', self.top_p)
        system_message = kwargs.get('system_message', self.system_message)
        
        # Prepare messages for Llama-3.3-70B Instruct
        messages = [
            SystemMessage(content=system_message),
            UserMessage(content=prompt)
        ]
        
        # Execute with retry logic
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Attempting Llama-3.3-70B API call (attempt {attempt + 1}/{self.max_retries + 1})")
                
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
                    logger.debug(f"Successfully received Llama-3.3-70B response ({len(content)} chars)")
                    return content
                else:
                    raise AzureLlamaClientError("Empty response received from Llama-3.3-70B model")
                    
            except HttpResponseError as e:
                error_code = getattr(e, 'status_code', 'Unknown')
                
                # Don't retry certain errors
                if error_code in [401, 403, 404]:
                    logger.error(f"Non-retryable Llama-3.3-70B error [{error_code}]: {e}")
                    raise AzureLlamaClientError(f"HTTP error [{error_code}]: {e}")
                
                # Retry other errors
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Retryable Llama-3.3-70B error on attempt {attempt + 1} [{error_code}]: {e}. "
                                 f"Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"Max retries exceeded. Last error [{error_code}]: {e}")
                    raise AzureLlamaClientError(f"Request failed after {self.max_retries} retries [{error_code}]: {e}")
                    
            except Exception as e:
                # Unexpected errors
                logger.error(f"Unexpected Llama-3.3-70B error: {e}")
                raise AzureLlamaClientError(f"Unexpected error: {e}")
        
        raise AzureLlamaClientError("Unexpected error: retry loop completed without result")
    
    async def close(self):
        """Close the client connection."""
        try:
            await self.client.close()
            logger.info("Azure Llama-3.3-70B client connection closed")
        except Exception as e:
            logger.warning(f"Error closing Llama-3.3-70B client: {e}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Convenience factory function
def create_llama_client(**kwargs) -> AzureLlamaClient:
    """Create a client for Llama-3.3-70B Instruct model."""
    return AzureLlamaClient(**kwargs)


# Test function for validation
async def test_llama_client():
    """Test function to validate Llama-3.3-70B client setup."""
    print("Testing Azure Llama-3.3-70B Instruct client...")
    
    try:
        client = create_llama_client()
        response = await client.query("Hello, how are you?")
        print(f"✅ Llama-3.3-70B: {response[:100]}...")
        await client.close()
    except Exception as e:
        print(f"❌ Llama-3.3-70B failed: {e}")


if __name__ == "__main__":
    # Run test
    asyncio.run(test_llama_client())
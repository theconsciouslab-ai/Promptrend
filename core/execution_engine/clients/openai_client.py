"""
OpenAI Client Wrapper

This module provides a wrapper for the OpenAI API to be used in the LLM vulnerability
testing framework. It handles Chat Completions API calls with proper error handling,
timeout management, and configuration flexibility.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import json

try:
    from openai import AsyncOpenAI
    from openai.types.chat import ChatCompletion
    from openai import OpenAIError, APITimeoutError, RateLimitError, APIError
except ImportError:
    raise ImportError(
        "OpenAI library not installed. Please install with: pip install openai"
    )


# Custom exceptions for better error handling
class OpenAIClientError(Exception):
    """Base exception for OpenAI client errors."""
    pass


class OpenAIConfigurationError(OpenAIClientError):
    """Raised when there's a configuration issue."""
    pass


class OpenAIAPIError(OpenAIClientError):
    """Raised when the OpenAI API returns an error."""
    pass


class OpenAITimeoutError(OpenAIClientError):
    """Raised when a request times out."""
    pass


class OpenAIRateLimitError(OpenAIClientError):
    """Raised when rate limit is exceeded."""
    pass


@dataclass
class OpenAIConfig:
    """Configuration for OpenAI client."""
    model: str
    api_key: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    base_url: Optional[str] = None
    organization: Optional[str] = None
    timeout: float = 30.0
    max_retries: int = 3
    system_message: str = "You are a helpful AI assistant."


class OpenAIClient:
    """
    OpenAI API client wrapper for LLM vulnerability testing.
    
    This client provides a simple interface to the OpenAI Chat Completions API
    with proper error handling, timeout management, and configuration flexibility.
    """
    
    def __init__(
        self,
        model: str,
        api_key: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        system_message: str = "You are a helpful AI assistant.",
        enable_logging: bool = True
    ):
        """
        Initialize the OpenAI client.
        
        Args:
            model: Model name (e.g., "gpt-4", "gpt-3.5-turbo")
            api_key: OpenAI API key
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens in response (None for model default)
            base_url: Custom base URL (for Azure OpenAI, OpenRouter, etc.)
            organization: OpenAI organization ID
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            system_message: System message to use for conversations
            enable_logging: Whether to enable detailed logging
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.system_message = system_message
        self.enable_logging = enable_logging
        
        # Validate required parameters
        if not model:
            raise OpenAIConfigurationError("Model name is required")
        if not api_key:
            raise OpenAIConfigurationError("API key is required")
        
        # Initialize the OpenAI client
        try:
            client_kwargs = {
                "api_key": api_key,
                "timeout": timeout,
                "max_retries": max_retries
            }
            
            if base_url:
                client_kwargs["base_url"] = base_url
            if organization:
                client_kwargs["organization"] = organization
                
            self.client = AsyncOpenAI(**client_kwargs)
            
        except Exception as e:
            raise OpenAIConfigurationError(f"Failed to initialize OpenAI client: {e}")
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if enable_logging and not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Statistics tracking
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens_used": 0,
            "rate_limit_errors": 0,
            "timeout_errors": 0
        }
        
        if self.enable_logging:
            self.logger.info(f"OpenAI client initialized for model: {self.model}")
    
    async def query(self, prompt: str, **kwargs) -> str:
        """
        Send a query to the OpenAI API and return the response.
        
        Args:
            prompt: The user prompt to send
            **kwargs: Optional overrides for temperature, max_tokens, etc.
            
        Returns:
            The model's response as a string
            
        Raises:
            OpenAIAPIError: If the API returns an error
            OpenAITimeoutError: If the request times out
            OpenAIRateLimitError: If rate limit is exceeded
            OpenAIClientError: For other client errors
        """
        if not prompt or not prompt.strip():
            if self.enable_logging:
                self.logger.warning("Empty prompt provided")
            return ""
        
        # Track request
        self.stats["total_requests"] += 1
        
        # Prepare request parameters
        request_params = self._prepare_request_params(prompt, **kwargs)
        
        if self.enable_logging:
            self.logger.debug(
                f"Sending request to {self.model} "
                f"(prompt length: {len(prompt)} chars, "
                f"temperature: {request_params.get('temperature')}, "
                f"max_tokens: {request_params.get('max_tokens')})"
            )
        
        try:
            # Make the API call
            response = await self.client.chat.completions.create(**request_params)
            
            # Extract the response content
            content = self._extract_response_content(response)
            
            # Update statistics
            self.stats["successful_requests"] += 1
            if hasattr(response, 'usage') and response.usage:
                self.stats["total_tokens_used"] += response.usage.total_tokens
            
            if self.enable_logging:
                self.logger.debug(
                    f"Received response from {self.model} "
                    f"(length: {len(content)} chars)"
                )
            
            return content
            
        except APITimeoutError as e:
            self.stats["failed_requests"] += 1
            self.stats["timeout_errors"] += 1
            error_msg = f"OpenAI API timeout for model {self.model}: {e}"
            
            if self.enable_logging:
                self.logger.error(error_msg)
            
            raise OpenAITimeoutError(error_msg) from e
            
        except RateLimitError as e:
            self.stats["failed_requests"] += 1
            self.stats["rate_limit_errors"] += 1
            error_msg = f"OpenAI rate limit exceeded for model {self.model}: {e}"
            
            if self.enable_logging:
                self.logger.warning(error_msg)
            
            raise OpenAIRateLimitError(error_msg) from e
            
        except APIError as e:
            self.stats["failed_requests"] += 1
            error_msg = f"OpenAI API error for model {self.model}: {e}"
            
            if self.enable_logging:
                self.logger.error(error_msg)
            
            raise OpenAIAPIError(error_msg) from e
            
        except OpenAIError as e:
            self.stats["failed_requests"] += 1
            error_msg = f"OpenAI client error for model {self.model}: {e}"
            
            if self.enable_logging:
                self.logger.error(error_msg)
            
            raise OpenAIClientError(error_msg) from e
            
        except Exception as e:
            self.stats["failed_requests"] += 1
            error_msg = f"Unexpected error for model {self.model}: {e}"
            
            if self.enable_logging:
                self.logger.error(error_msg)
            
            raise OpenAIClientError(error_msg) from e
    
    def _prepare_request_params(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Prepare the parameters for the OpenAI API request.
        
        Args:
            prompt: The user prompt
            **kwargs: Optional parameter overrides
            
        Returns:
            Dictionary of request parameters
        """
        # Build messages array
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": prompt}
        ]
        
        # Prepare base parameters
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
        }
        
        # Add optional parameters
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        
        # Add any additional parameters from kwargs
        additional_params = {
            "top_p", "frequency_penalty", "presence_penalty", 
            "stop", "logit_bias", "user", "response_format", "tools"
        }
        
        for param in additional_params:
            if param in kwargs:
                params[param] = kwargs[param]
        
        return params
    
    def _extract_response_content(self, response: ChatCompletion) -> str:
        """
        Extract content from the OpenAI API response.
        
        Args:
            response: The ChatCompletion response object
            
        Returns:
            The content string from the first choice
        """
        try:
            if not response.choices:
                if self.enable_logging:
                    self.logger.warning("No choices in API response")
                return ""
            
            first_choice = response.choices[0]
            if not hasattr(first_choice, 'message'):
                if self.enable_logging:
                    self.logger.warning("No message in first choice")
                return ""
            
            content = first_choice.message.content
            return content if content is not None else ""
            
        except (AttributeError, IndexError, KeyError) as e:
            if self.enable_logging:
                self.logger.error(f"Error extracting response content: {e}")
            return ""
    
    async def query_with_retries(self, prompt: str, retries: int = None, **kwargs) -> str:
        """
        Query with automatic retry on failure.
        
        Args:
            prompt: The user prompt
            retries: Number of retry attempts (defaults to client max_retries)
            **kwargs: Additional parameters
            
        Returns:
            The model's response as a string
        """
        max_attempts = (retries if retries is not None else self.max_retries) + 1
        last_exception = None
        
        for attempt in range(max_attempts):
            try:
                return await self.query(prompt, **kwargs)
            except (OpenAIRateLimitError, OpenAITimeoutError) as e:
                last_exception = e
                if attempt < max_attempts - 1:
                    # Exponential backoff for retries
                    wait_time = (2 ** attempt) * 1.0  # 1s, 2s, 4s, etc.
                    if self.enable_logging:
                        self.logger.info(f"Retrying in {wait_time}s (attempt {attempt + 1}/{max_attempts})")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise
            except OpenAIAPIError as e:
                # Don't retry on API errors (usually client issues)
                raise
        
        # If we get here, all retries failed
        if last_exception:
            raise last_exception
        else:
            raise OpenAIClientError("All retry attempts failed")
    
    async def test_connection(self) -> bool:
        """
        Test the connection to the OpenAI API.
        
        Returns:
            True if connection is working, False otherwise
        """
        try:
            response = await self.query("Hello, this is a test.", max_tokens=10)
            return len(response.strip()) > 0
        except Exception as e:
            if self.enable_logging:
                self.logger.error(f"Connection test failed: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get client usage statistics.
        
        Returns:
            Dictionary containing usage statistics
        """
        stats = self.stats.copy()
        if stats["total_requests"] > 0:
            stats["success_rate"] = stats["successful_requests"] / stats["total_requests"]
            stats["failure_rate"] = stats["failed_requests"] / stats["total_requests"]
        else:
            stats["success_rate"] = 0.0
            stats["failure_rate"] = 0.0
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens_used": 0,
            "rate_limit_errors": 0,
            "timeout_errors": 0
        }
        
        if self.enable_logging:
            self.logger.info("Statistics reset")
    
    def __repr__(self) -> str:
        """String representation of the client."""
        return f"OpenAIClient(model='{self.model}', temperature={self.temperature})"


# Convenience functions for common configurations
def create_gpt4_client(api_key: str, **kwargs) -> OpenAIClient:
    """
    Create a GPT-4 client with sensible defaults.
    
    Args:
        api_key: OpenAI API key
        **kwargs: Additional client parameters
        
    Returns:
        Configured OpenAIClient instance
    """
    return OpenAIClient(
        model="gpt-4",
        api_key=api_key,
        temperature=kwargs.get("temperature", 0.7),
        max_tokens=kwargs.get("max_tokens", 2048),
        **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]}
    )


def create_gpt35_turbo_client(api_key: str, **kwargs) -> OpenAIClient:
    """
    Create a GPT-3.5-turbo client with sensible defaults.
    
    Args:
        api_key: OpenAI API key
        **kwargs: Additional client parameters
        
    Returns:
        Configured OpenAIClient instance
    """
    return OpenAIClient(
        model="gpt-3.5-turbo",
        api_key=api_key,
        temperature=kwargs.get("temperature", 0.7),
        max_tokens=kwargs.get("max_tokens", 1024),
        **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]}
    )


def create_azure_openai_client(
    api_key: str,
    base_url: str,
    model: str,
    **kwargs
) -> OpenAIClient:
    """
    Create an Azure OpenAI client.
    
    Args:
        api_key: Azure OpenAI API key
        base_url: Azure OpenAI endpoint URL
        model: Deployment name
        **kwargs: Additional client parameters
        
    Returns:
        Configured OpenAIClient instance
    """
    return OpenAIClient(
        model=model,
        api_key=api_key,
        base_url=base_url,
        **kwargs
    )


# Example usage and configuration loading
def load_client_from_env() -> OpenAIClient:
    """
    Load OpenAI client from environment variables.
    
    Expected environment variables:
    - OPENAI_API_KEY: Your OpenAI API key
    - OPENAI_MODEL: Model name (default: gpt-3.5-turbo)
    - OPENAI_TEMPERATURE: Temperature setting (default: 0.7)
    - OPENAI_MAX_TOKENS: Max tokens (optional)
    - OPENAI_BASE_URL: Custom base URL (optional)
    
    Returns:
        Configured OpenAIClient instance
    """
    import os
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise OpenAIConfigurationError("OPENAI_API_KEY environment variable is required")
    
    model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    max_tokens = os.getenv("OPENAI_MAX_TOKENS")
    base_url = os.getenv("OPENAI_BASE_URL")
    
    kwargs = {
        "model": model,
        "api_key": api_key,
        "temperature": temperature
    }
    
    if max_tokens:
        kwargs["max_tokens"] = int(max_tokens)
    if base_url:
        kwargs["base_url"] = base_url
    
    return OpenAIClient(**kwargs)

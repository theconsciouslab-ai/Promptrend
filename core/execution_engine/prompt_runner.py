"""
Prompt Runner Module

This module provides the core functionality for executing prompts across different
LLM model clients with proper error handling, timeout management, and logging.
"""

import asyncio
import logging
from core.execution_engine.model_client import ModelClient
from datetime import datetime
from typing import Optional, Dict, Any

# Custom exceptions for better error handling
class PromptExecutionError(Exception):
    """Base exception for prompt execution errors."""
    pass


class ModelTimeoutError(PromptExecutionError):
    """Raised when a model query times out."""
    pass


class ModelAPIError(PromptExecutionError):
    """Raised when a model API call fails."""
    pass


class ModelClientError(PromptExecutionError):
    """Raised when there's an issue with the model client itself."""
    pass





class PromptRunner:
    """
    Handles execution of prompts across different model clients.
    
    Features:
    - Timeout management per client
    - Comprehensive error handling
    - Detailed logging
    - Retry logic (optional)
    - Performance metrics
    """
    
    def __init__(self, enable_logging: bool = True, log_level: str = "INFO"):
        """
        Initialize the prompt runner.
        
        Args:
            enable_logging: Whether to enable detailed logging
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.enable_logging = enable_logging
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if enable_logging and not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Metrics tracking
        self.execution_stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "timeout_calls": 0,
            "avg_response_time": 0.0
        }
    
    async def run_prompt(
        self, 
        model_client: ModelClient, 
        prompt: str,
        override_timeout: Optional[float] = None,
        retry_count: int = 0,
        return_empty_on_error: bool = True
    ) -> str:
        """
        Execute a prompt using the specified model client.
        
        Args:
            model_client: The ModelClient instance containing client and metadata
            prompt: The prompt string to send to the model
            override_timeout: Optional timeout override for this specific call
            retry_count: Number of retries on failure (default: 0)
            return_empty_on_error: If True, return empty string on error; if False, raise exception
            
        Returns:
            The model's response as a string, or empty string on error (if return_empty_on_error=True)
            
        Raises:
            ModelTimeoutError: If the request times out
            ModelAPIError: If the API call fails
            ModelClientError: If there's an issue with the client
            PromptExecutionError: For other execution errors
        """
        if not isinstance(model_client, ModelClient):
            raise ModelClientError(f"Expected ModelClient, got {type(model_client)}")
        
        if not prompt or not prompt.strip():
            if self.enable_logging:
                self.logger.warning(f"Empty prompt provided to {model_client.name}")
            return ""
        
        # Determine timeout to use
        timeout = override_timeout if override_timeout is not None else model_client.timeout
        
        start_time = datetime.utcnow()
        
        if self.enable_logging:
            self.logger.debug(
                f"Starting prompt execution for {model_client.name} "
                f"(timeout: {timeout}s, prompt length: {len(prompt)} chars)"
            )
        
        # Track metrics
        self.execution_stats["total_calls"] += 1
        
        try:
            # Validate that the client has the required query method
            if not hasattr(model_client.client, 'query'):
                raise ModelClientError(
                    f"Model client {model_client.name} does not have a 'query' method"
                )
            
            if not callable(getattr(model_client.client, 'query')):
                raise ModelClientError(
                    f"Model client {model_client.name} 'query' attribute is not callable"
                )
            
            # Execute the prompt with timeout
            response = await asyncio.wait_for(
                model_client.client.query(prompt),
                timeout=timeout
            )
            
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Update metrics
            self.execution_stats["successful_calls"] += 1
            self._update_avg_response_time(execution_time)
            
            if self.enable_logging:
                self.logger.info(
                    f"Successfully executed prompt for {model_client.name} "
                    f"(time: {execution_time:.2f}s, response length: {len(str(response))} chars)"
                )
                self.logger.debug(f"Response preview: {str(response)[:100]}...")
            
            return str(response) if response is not None else ""
            
        except asyncio.TimeoutError:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            error_msg = f"Timeout after {timeout}s for {model_client.name}"
            
            self.execution_stats["failed_calls"] += 1
            self.execution_stats["timeout_calls"] += 1
            
            if self.enable_logging:
                self.logger.warning(f"{error_msg} (actual time: {execution_time:.2f}s)")
            
            if retry_count > 0:
                if self.enable_logging:
                    self.logger.info(f"Retrying {model_client.name} ({retry_count} retries left)")
                return await self.run_prompt(
                    model_client, prompt, override_timeout, retry_count - 1, return_empty_on_error
                )
            
            if return_empty_on_error:
                return ""
            else:
                raise ModelTimeoutError(error_msg)
        
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            error_str = str(e)
            
            self.execution_stats["failed_calls"] += 1
            
            if self.enable_logging:
                self.logger.error(f"API error for {model_client.name}: {error_str} (time: {execution_time:.2f}s)")
                self.logger.debug(f"Exception details: {type(e).__name__}: {e}")

            # Special handling for Azure content filtering
            if "ResponsibleAIPolicyViolation" in error_str:
                raise RuntimeError("Azure Content Filter triggered") from e

            if retry_count > 0:
                if self.enable_logging:
                    self.logger.info(f"Retrying {model_client.name} ({retry_count} retries left)")
                return await self.run_prompt(
                    model_client, prompt, override_timeout, retry_count - 1, return_empty_on_error
                )

            if return_empty_on_error:
                return ""
            else:
                if "timeout" in error_str.lower():
                    raise ModelTimeoutError(error_str) from e
                elif "api" in error_str.lower() or "request" in error_str.lower():
                    raise ModelAPIError(error_str) from e
                else:
                    raise PromptExecutionError(error_str) from e

    
    def _update_avg_response_time(self, execution_time: float) -> None:
        """Update the average response time metric."""
        total_successful = self.execution_stats["successful_calls"]
        if total_successful == 1:
            self.execution_stats["avg_response_time"] = execution_time
        else:
            current_avg = self.execution_stats["avg_response_time"]
            self.execution_stats["avg_response_time"] = (
                (current_avg * (total_successful - 1) + execution_time) / total_successful
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get execution statistics.
        
        Returns:
            Dictionary containing execution metrics
        """
        stats = self.execution_stats.copy()
        if stats["total_calls"] > 0:
            stats["success_rate"] = stats["successful_calls"] / stats["total_calls"]
            stats["failure_rate"] = stats["failed_calls"] / stats["total_calls"]
            stats["timeout_rate"] = stats["timeout_calls"] / stats["total_calls"]
        else:
            stats["success_rate"] = 0.0
            stats["failure_rate"] = 0.0
            stats["timeout_rate"] = 0.0
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset all execution statistics."""
        self.execution_stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "timeout_calls": 0,
            "avg_response_time": 0.0
        }
        
        if self.enable_logging:
            self.logger.info("Execution statistics reset")
    
    async def validate_client(self, model_client: ModelClient, test_prompt: str = "Hello") -> bool:
        """
        Validate that a model client is working properly.
        
        Args:
            model_client: The ModelClient to validate
            test_prompt: Simple prompt to test with
            
        Returns:
            True if client is working, False otherwise
        """
        try:
            response = await self.run_prompt(
                model_client, 
                test_prompt, 
                override_timeout=min(model_client.timeout, 10.0),
                return_empty_on_error=False
            )
            return len(response.strip()) > 0
        except Exception as e:
            if self.enable_logging:
                self.logger.warning(f"Client validation failed for {model_client.name}: {e}")
            return False


# Convenience functions for common use cases
async def run_prompt(
    model_client: ModelClient, 
    prompt: str,
    **kwargs
) -> str:
    """
    Convenience function to run a single prompt.
    
    Args:
        model_client: The ModelClient instance
        prompt: The prompt to execute
        **kwargs: Additional arguments passed to PromptRunner.run_prompt
        
    Returns:
        The model's response as a string
    """
    runner = PromptRunner()
    return await runner.run_prompt(model_client, prompt, **kwargs)


async def run_prompt_with_retry(
    model_client: ModelClient,
    prompt: str,
    max_retries: int = 3,
    **kwargs
) -> str:
    """
    Convenience function to run a prompt with automatic retries.
    
    Args:
        model_client: The ModelClient instance
        prompt: The prompt to execute
        max_retries: Maximum number of retry attempts
        **kwargs: Additional arguments passed to PromptRunner.run_prompt
        
    Returns:
        The model's response as a string
    """
    runner = PromptRunner()
    return await runner.run_prompt(model_client, prompt, retry_count=max_retries, **kwargs)


async def validate_multiple_clients(
    model_clients: list[ModelClient],
    test_prompt: str = "Hello"
) -> Dict[str, bool]:
    """
    Validate multiple model clients concurrently.
    
    Args:
        model_clients: List of ModelClient instances to validate
        test_prompt: Simple prompt to test with
        
    Returns:
        Dictionary mapping client names to validation results
    """
    runner = PromptRunner()
    
    async def validate_single(client: ModelClient) -> tuple[str, bool]:
        result = await runner.validate_client(client, test_prompt)
        return client.name, result
    
    tasks = [validate_single(client) for client in model_clients]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    validation_results = {}
    for result in results:
        if isinstance(result, Exception):
            # Handle any unexpected errors in validation
            continue
        name, is_valid = result
        validation_results[name] = is_valid
    
    return validation_results

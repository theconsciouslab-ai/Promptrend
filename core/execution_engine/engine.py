"""
LLM Vulnerability Testing Engine

This module provides the core execution engine for benchmarking LLM vulnerability
prompts across different models with concurrent execution and comprehensive error handling.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from core.execution_engine.model_client import ModelClient


from .evaluator import evaluate
from .prompt_runner import PromptRunner
from ..schemas import ExecutionResult




class ExecutionEngine:
    """
    Core execution engine for running vulnerability prompts across multiple LLM models.
    
    Features:
    - Concurrent execution across models
    - Comprehensive error handling
    - Configurable timeouts
    - Detailed logging
    - Rate limiting support
    """
    
    def __init__(
        self,
        model_clients: List[ModelClient],
        max_concurrent: int = 5,
        default_timeout: float = 120.0,
        rate_limit_delay: float = 0.1
    ):
        """
        Initialize the execution engine.
        
        Args:
            model_clients: List of ModelClient instances with name and client
            max_concurrent: Maximum number of concurrent model requests
            default_timeout: Default timeout for model requests in seconds
            rate_limit_delay: Delay between requests to avoid rate limiting
        """
        self.model_clients = {client.name: client for client in model_clients}
        self.max_concurrent = max_concurrent
        self.default_timeout = default_timeout
        self.rate_limit_delay = rate_limit_delay
        self.prompt_runner = PromptRunner()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    async def execute_prompt(
        self,
        prompt: str,
        models: Optional[List[str]] = None,
        timeout: Optional[float] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Execute a vulnerability prompt across specified models.
        
        Args:
            prompt: The prompt string to test
            models: List of model names to test (if None, tests all configured models)
            timeout: Override timeout for this execution
            
        Returns:
            Dictionary mapping model names to execution results
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        # Determine which models to test
        target_models = models or list(self.model_clients.keys())
        missing_models = set(target_models) - set(self.model_clients.keys())
        if missing_models:
            raise ValueError(f"Unknown models: {missing_models}")
        
        self.logger.info(f"Starting execution for {len(target_models)} models")
        self.logger.debug(f"Target models: {target_models}")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # Create tasks for each model
        tasks = []
        for model_name in target_models:
            model_client = self.model_clients[model_name]
            task = asyncio.create_task(
                self._execute_single_model(
                    prompt=prompt,
                    model_name=model_name,
                    model_client=model_client,
                    semaphore=semaphore,
                    timeout=timeout or model_client.timeout or self.default_timeout
                )
            )
            tasks.append((model_name, task))
        
        # Execute all tasks concurrently
        results = {}
        completed_tasks = await asyncio.gather(
            *[task for _, task in tasks],
            return_exceptions=True
        )
        
        # Process results
        for (model_name, _), result in zip(tasks, completed_tasks):
            if isinstance(result, Exception):
                self.logger.error(f"Task failed for {model_name}: {result}")
                results[model_name] = self._create_error_result(str(result))
            else:
                results[model_name] = result
        
        self.logger.info(f"Execution completed for {len(results)} models")
        return results
    
    async def _execute_single_model(
        self,
        prompt: str,
        model_name: str,
        model_client: ModelClient,
        semaphore: asyncio.Semaphore,
        timeout: float
    ) -> Dict[str, Any]:
        """
        Execute prompt for a single model with error handling.
        
        Args:
            prompt: The prompt to execute
            model_name: Name of the model
            model_client: The model client wrapper
            semaphore: Semaphore for concurrency control
            timeout: Timeout for this execution
            
        Returns:
            Dictionary with execution result
        """
        start_time = datetime.utcnow()
        
        async with semaphore:
            try:
                self.logger.debug(f"Starting execution for {model_name}")
                
                # Add rate limiting delay
                if self.rate_limit_delay > 0:
                    await asyncio.sleep(self.rate_limit_delay)
                
                # Execute the prompt using prompt_runner
                # IMPORTANT: Pass the ModelClient wrapper, not the raw client
                response = await asyncio.wait_for(
                    self.prompt_runner.run_prompt(model_client, prompt),
                    timeout=timeout
                )
                
                # Evaluate the response
                is_success = evaluate(response)
                
                result = {
                    "success": is_success,
                    "response": response,
                    "timestamp": start_time.isoformat(),
                    "execution_time": (datetime.utcnow() - start_time).total_seconds(),
                    "model": model_name,
                    "error": None
                }
                
                self.logger.debug(
                    f"Completed {model_name}: success={is_success}, "
                    f"time={result['execution_time']:.2f}s"
                )
                
                return result
                
            except asyncio.TimeoutError:
                error_msg = f"Timeout after {timeout}s"
                self.logger.warning(f"{model_name}: {error_msg}")
                return self._create_error_result(error_msg, start_time, model_name)
                
            except Exception as e:
                error_msg = str(e)
                self.logger.error(f"{model_name}: Execution failed: {error_msg}")
                
                if "ResponsibleAIPolicyViolation" in error_msg:
                    filter_flags = {
                        "jailbreak": "'jailbreak': {'filtered': True" in error_msg,
                        "self_harm": "'self_harm': {'filtered': True" in error_msg,
                        "sexual": "'sexual': {'filtered': True" in error_msg,
                        "violence": "'violence': {'filtered': True" in error_msg,
                        "hate": "'hate': {'filtered': True" in error_msg,
                    }

                    return {
                        "success": False,
                        "response": None,
                        "timestamp": start_time.isoformat(),
                        "execution_time": (datetime.utcnow() - start_time).total_seconds(),
                        "model": model_name,
                        "error": "Azure Content Filter triggered",
                        "filter_type": filter_flags
                    }
                
                return self._create_error_result(error_msg, start_time, model_name)


    
    def _create_error_result(
        self,
        error_message: str,
        start_time: Optional[datetime] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a standardized error result."""
        if start_time is None:
            start_time = datetime.utcnow()
        
        return {
            "success": False,
            "response": None,
            "timestamp": start_time.isoformat(),
            "execution_time": (datetime.utcnow() - start_time).total_seconds(),
            "model": model_name,
            "error": error_message
        }
    
    async def execute_batch(
        self,
        prompts: List[str],
        models: Optional[List[str]] = None,
        timeout: Optional[float] = None
    ) -> List[Dict[str, Dict[str, Any]]]:
        """
        Execute multiple prompts across models.
        
        Args:
            prompts: List of prompts to execute
            models: List of model names to test
            timeout: Override timeout for executions
            
        Returns:
            List of result dictionaries, one per prompt
        """
        if not prompts:
            raise ValueError("Prompts list cannot be empty")
        
        self.logger.info(f"Starting batch execution for {len(prompts)} prompts")
        
        tasks = []
        for i, prompt in enumerate(prompts):
            task = asyncio.create_task(
                self.execute_prompt(prompt, models, timeout)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions in batch processing
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Batch item {i} failed: {result}")
                # Create error results for all models
                target_models = models or list(self.model_clients.keys())
                error_result = {
                    model: self._create_error_result(f"Batch execution failed: {result}")
                    for model in target_models
                }
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        self.logger.info(f"Batch execution completed for {len(processed_results)} prompts")
        return processed_results
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics about configured models."""
        return {
            "total_models": len(self.model_clients),
            "model_names": list(self.model_clients.keys()),
            "max_concurrent": self.max_concurrent,
            "default_timeout": self.default_timeout,
            "rate_limit_delay": self.rate_limit_delay
        }
    
    async def health_check(
        self,
        test_prompt: str = "Hello, how are you?",
        timeout: float = 10.0
    ) -> Dict[str, bool]:
        """
        Perform a health check on all configured models.
        
        Args:
            test_prompt: Simple prompt to test model availability
            timeout: Timeout for health check requests
            
        Returns:
            Dictionary mapping model names to health status
        """
        self.logger.info("Starting health check for all models")
        
        health_results = {}
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def check_model(model_name: str, model_client: ModelClient) -> Tuple[str, bool]:
            async with semaphore:
                try:
                    await asyncio.wait_for(
                        self.prompt_runner.run_prompt(model_client, test_prompt),
                        timeout=timeout
                    )
                    return model_name, True
                except Exception as e:
                    self.logger.warning(f"Health check failed for {model_name}: {e}")
                    return model_name, False
        
        tasks = [
            check_model(name, client)
            for name, client in self.model_clients.items()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Health check task failed: {result}")
            else:
                model_name, is_healthy = result
                health_results[model_name] = is_healthy
        
        healthy_count = sum(health_results.values())
        self.logger.info(f"Health check completed: {healthy_count}/{len(health_results)} models healthy")
        
        return health_results


# Convenience function for simple usage
async def execute_vulnerability_test(
    prompt: str,
    model_clients: List[ModelClient],
    **kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Convenience function to execute a single vulnerability test.
    
    Args:
        prompt: The vulnerability prompt to test
        model_clients: List of model clients to test against
        **kwargs: Additional arguments passed to ExecutionEngine
        
    Returns:
        Dictionary mapping model names to execution results
    """
    engine = ExecutionEngine(model_clients, **kwargs)
    return await engine.execute_prompt(prompt)
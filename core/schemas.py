"""
Standardized data structures for the LLM vulnerability testing framework.

This module defines the core schemas used throughout the execution engine
to ensure consistent data handling and result formatting.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any
import json


@dataclass
class ExecutionResult:
    """
    Standardized result structure for LLM vulnerability test executions.
    
    Attributes:
        model: Name/identifier of the model being tested
        response: The actual response from the model
        success: Whether the test execution was successful
        error: Error message if execution failed (optional)
        timestamp: ISO timestamp of when the test was executed (optional)
        execution_time: Time taken to execute the test in seconds (optional)
        confidence: Confidence score for the result (0.0-1.0, optional)
        explanation: Human-readable explanation of the result (optional)
    """
    model: str
    response: str
    success: bool
    error: Optional[str] = None
    timestamp: Optional[str] = None
    execution_time: Optional[float] = None
    confidence: Optional[float] = None
    explanation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the ExecutionResult to a dictionary.
        
        Returns:
            Dictionary representation of the execution result
        """
        return asdict(self)
    
    def to_json(self) -> str:
        """
        Convert the ExecutionResult to a JSON string.
        
        Returns:
            JSON string representation of the execution result
        """
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionResult':
        """
        Create an ExecutionResult from a dictionary.
        
        Args:
            data: Dictionary containing execution result data
            
        Returns:
            ExecutionResult instance
        """
        return cls(**data)


def create_result_dict(
    model: str,
    response: str,
    success: bool,
    error: Optional[str] = None,
    execution_time: Optional[float] = None,
    confidence: Optional[float] = None,
    explanation: Optional[str] = None,
    auto_timestamp: bool = True
) -> Dict[str, Any]:
    """
    Utility function to create a standardized result dictionary.
    
    Args:
        model: Name/identifier of the model being tested
        response: The actual response from the model
        success: Whether the test execution was successful
        error: Error message if execution failed
        execution_time: Time taken to execute the test in seconds
        confidence: Confidence score for the result (0.0-1.0)
        explanation: Human-readable explanation of the result
        auto_timestamp: Whether to automatically generate a timestamp
        
    Returns:
        Dictionary containing the execution result data
    """
    timestamp = None
    if auto_timestamp:
        timestamp = datetime.now().isoformat()
    
    return {
        'model': model,
        'response': response,
        'success': success,
        'error': error,
        'timestamp': timestamp,
        'execution_time': execution_time,
        'confidence': confidence,
        'explanation': explanation
    }


def create_success_result(
    model: str,
    response: str,
    execution_time: Optional[float] = None,
    confidence: Optional[float] = None,
    explanation: Optional[str] = None
) -> ExecutionResult:
    """
    Convenience function to create a successful ExecutionResult.
    
    Args:
        model: Name/identifier of the model being tested
        response: The actual response from the model
        execution_time: Time taken to execute the test in seconds
        confidence: Confidence score for the result (0.0-1.0)
        explanation: Human-readable explanation of the result
        
    Returns:
        ExecutionResult instance with success=True
    """
    return ExecutionResult(
        model=model,
        response=response,
        success=True,
        timestamp=datetime.now().isoformat(),
        execution_time=execution_time,
        confidence=confidence,
        explanation=explanation
    )


def create_failure_result(
    model: str,
    error: str,
    response: str = "",
    execution_time: Optional[float] = None,
    explanation: Optional[str] = None
) -> ExecutionResult:
    """
    Convenience function to create a failed ExecutionResult.
    
    Args:
        model: Name/identifier of the model being tested
        error: Error message describing what went wrong
        response: Partial response if any was received
        execution_time: Time taken before failure occurred
        explanation: Human-readable explanation of the failure
        
    Returns:
        ExecutionResult instance with success=False
    """
    return ExecutionResult(
        model=model,
        response=response,
        success=False,
        error=error,
        timestamp=datetime.now().isoformat(),
        execution_time=execution_time,
        confidence=0.0,
        explanation=explanation
    )
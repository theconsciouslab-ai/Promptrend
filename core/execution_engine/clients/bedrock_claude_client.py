"""
AWS Bedrock Claude client wrapper for LLM benchmarking system.

This module provides an async client for interacting with AWS Bedrock's 
Claude models with proper error handling and retry logic.
"""

import asyncio
import logging
import json
from typing import Optional, Dict, Any
import boto3
from botocore.exceptions import (
    BotoCoreError, 
    ClientError, 
    NoCredentialsError,
    PartialCredentialsError,
    ConnectTimeoutError,
    ReadTimeoutError
)

import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class BedrockClaudeClientError(Exception):
    """Custom exception for Bedrock Claude client errors."""
    pass


class BedrockClaudeClient:
    """
    Async client wrapper for AWS Bedrock Claude models.
    
    Provides a simple interface for querying Claude models via AWS Bedrock with
    built-in retry logic, timeout handling, and error management.
    """
    
    def __init__(
        self,
        model_id: str,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: str = "eu-central-1",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        top_p: float = 0.999,
        timeout: float = 90.0,
        max_retries: int = 5,
        retry_delay: float = 3.0,
        inference_profile_arn: Optional[str] = None
    ):
        """
        Initialize the Bedrock Claude client.
        
        Args:
            model_id: Claude model ID (e.g., "anthropic.claude-3-5-sonnet-20240620-v1:0")
            aws_access_key_id: AWS access key ID (or from env BEDROCK_ACCESS_KEY_ID)
            aws_secret_access_key: AWS secret key (or from env BEDROCK_SECRET_ACCESS_KEY)
            region_name: AWS region (default: eu-central-1)
            temperature: Model temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            retry_delay: Base delay between retries
            inference_profile_arn: Required for Claude 3.7 Sonnet
        """
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.inference_profile_arn = inference_profile_arn
        
        # Load credentials from environment if not passed
        self.aws_access_key_id = aws_access_key_id or os.getenv("BEDROCK_ACCESS_KEY_ID")
        self.aws_secret_access_key = aws_secret_access_key or os.getenv("BEDROCK_SECRET_ACCESS_KEY")
        self.region_name = region_name
        
        if not self.aws_access_key_id or not self.aws_secret_access_key:
            raise BedrockClaudeClientError("Missing AWS credentials in environment variables")
        
        # Initialize Bedrock client
        try:
            self.client = boto3.client(
                service_name='bedrock-runtime',
                region_name=self.region_name,
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key
            )
        except Exception as e:
            raise BedrockClaudeClientError(f"Failed to initialize Bedrock client: {e}")
        
        # Determine effective model ID (use inference profile ARN if provided)
        self.effective_model_id = self.inference_profile_arn or self.model_id
        
        logger.info(f"Initialized Bedrock Claude client for model: {self.model_id}")
        if self.inference_profile_arn:
            logger.info(f"Using inference profile ARN: {self.inference_profile_arn}")
    
    async def query(self, prompt: str, **kwargs) -> str:
        """
        Query the Claude model with the given prompt.
        
        Args:
            prompt: User prompt/question
            **kwargs: Additional parameters to override defaults
                - temperature: Override default temperature
                - max_tokens: Override default max_tokens
                - top_p: Override default top_p
                - system_message: System message (optional)
        
        Returns:
            str: Model response content
            
        Raises:
            BedrockClaudeClientError: On API errors or failures
        """
        # Extract parameters with fallbacks to instance defaults
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        top_p = kwargs.get('top_p', self.top_p)
        system_message = kwargs.get('system_message')
        
        # Prepare messages for Claude
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
        
        # Prepare request body
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "messages": messages
        }
        
        # Add system message if provided
        if system_message:
            request_body["system"] = system_message
        
        # Execute with retry logic
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Attempting Bedrock API call (attempt {attempt + 1}/{self.max_retries + 1})")
                
                # Run the synchronous boto3 call in a thread pool to make it async
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.invoke_model(
                        modelId=self.effective_model_id,
                        contentType="application/json",
                        accept="application/json",
                        body=json.dumps(request_body)
                    )
                )
                
                # Parse response
                response_body = json.loads(response['body'].read())
                
                # Extract content from Claude response format
                if 'content' in response_body:
                    content_parts = response_body['content']
                    if content_parts and isinstance(content_parts, list):
                        text_content = []
                        for part in content_parts:
                            if part.get('type') == 'text':
                                text_content.append(part.get('text', ''))
                        
                        if text_content:
                            full_response = '\n'.join(text_content)
                            logger.debug(f"Successfully received response ({len(full_response)} chars)")
                            return full_response
                        else:
                            raise BedrockClaudeClientError("No text content in response")
                    else:
                        raise BedrockClaudeClientError("Invalid content format in response")
                else:
                    raise BedrockClaudeClientError("No content in response")
                    
            except NoCredentialsError as e:
                # Don't retry credential errors
                logger.error(f"AWS credentials error: {e}")
                raise BedrockClaudeClientError(f"AWS credentials error: {e}")
                
            except PartialCredentialsError as e:
                # Don't retry partial credential errors
                logger.error(f"Partial AWS credentials error: {e}")
                raise BedrockClaudeClientError(f"Partial AWS credentials error: {e}")
                
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                error_message = e.response.get('Error', {}).get('Message', str(e))
                
                # Don't retry certain client errors
                if error_code in ['ValidationException', 'AccessDeniedException', 'UnauthorizedOperation']:
                    logger.error(f"Non-retryable client error [{error_code}]: {error_message}")
                    raise BedrockClaudeClientError(f"Client error [{error_code}]: {error_message}")
                
                # Retry throttling and temporary errors
                if error_code in ['ThrottlingException', 'ServiceUnavailableException', 'InternalServerException']:
                    if attempt < self.max_retries:
                        delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"Retryable error on attempt {attempt + 1} [{error_code}]: {error_message}. "
                                     f"Retrying in {delay:.1f}s...")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.error(f"Max retries exceeded. Last error [{error_code}]: {error_message}")
                        raise BedrockClaudeClientError(f"Request failed after {self.max_retries} retries [{error_code}]: {error_message}")
                else:
                    # Unknown client error
                    logger.error(f"Unknown client error [{error_code}]: {error_message}")
                    raise BedrockClaudeClientError(f"Client error [{error_code}]: {error_message}")
                    
            except (ConnectTimeoutError, ReadTimeoutError) as e:
                # Retry timeout errors
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Timeout error on attempt {attempt + 1}: {e}. "
                                 f"Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"Max retries exceeded. Last error: {e}")
                    raise BedrockClaudeClientError(f"Request timed out after {self.max_retries} retries: {e}")
                    
            except BotoCoreError as e:
                # Generic boto3 errors
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Boto3 error on attempt {attempt + 1}: {e}. "
                                 f"Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"Max retries exceeded. Last error: {e}")
                    raise BedrockClaudeClientError(f"Boto3 error after {self.max_retries} retries: {e}")
                    
            except Exception as e:
                # Unexpected errors
                logger.error(f"Unexpected error: {e}")
                raise BedrockClaudeClientError(f"Unexpected error: {e}")
        
        # This should never be reached due to the loop structure
        raise BedrockClaudeClientError("Unexpected error: retry loop completed without result")
    
    def close(self):
        """Close the client connection (boto3 doesn't require explicit closing)."""
        logger.info("Bedrock Claude client connection closed")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.close()


# Convenience factory functions for different Claude models

def create_claude_35_sonnet_client(**kwargs) -> BedrockClaudeClient:
    """Create a client for Claude 3.5 Sonnet."""
    return BedrockClaudeClient(
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        **kwargs
    )


def create_claude_37_sonnet_client(inference_profile_arn: str, **kwargs) -> BedrockClaudeClient:
    """Create a client for Claude 3.7 Sonnet (requires inference profile ARN)."""
    return BedrockClaudeClient(
        model_id="claude-3-7-sonnet",  # Placeholder, ARN will be used
        inference_profile_arn=inference_profile_arn,
        **kwargs
    )


def create_claude_haiku_client(**kwargs) -> BedrockClaudeClient:
    """Create a client for Claude 3 Haiku."""
    return BedrockClaudeClient(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        **kwargs
    )


def create_claude_4_sonnet_client(**kwargs) -> BedrockClaudeClient:
    """
    Create a client for Claude 4 Sonnet using US region.
    
    Note: Claude 4 models are ONLY available from US source regions.
    This function forces us-east-1 regardless of your default region.
    """
    # Force US region for Claude 4 - required by AWS
    kwargs['region_name'] = 'us-east-1'
    print(f"   ℹ️  Using us-east-1 for Claude 4 Sonnet (required by AWS - not available from EU)")
    
    return BedrockClaudeClient(
        model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
        **kwargs
    )


def create_claude_4_opus_client(**kwargs) -> BedrockClaudeClient:
    """
    Create a client for Claude 4 Opus using US region.
    
    Note: Claude 4 models are ONLY available from US source regions.
    This function forces us-east-1 regardless of your default region.
    """
    # Force US region for Claude 4 - required by AWS
    kwargs['region_name'] = 'us-east-1'
    print(f"   ℹ️  Using us-east-1 for Claude 4 Opus (required by AWS - not available from EU)")
    
    return BedrockClaudeClient(
        model_id="us.anthropic.claude-opus-4-20250514-v1:0",
        **kwargs
    )


def create_claude_4_sonnet_client_us_west(**kwargs) -> BedrockClaudeClient:
    """Create a client for Claude 4 Sonnet using US West region (alternative)."""
    kwargs['region_name'] = 'us-west-2'
    return BedrockClaudeClient(
        model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
        **kwargs
    )


def create_claude_4_opus_client_us_west(**kwargs) -> BedrockClaudeClient:
    """Create a client for Claude 4 Opus using US West region (alternative)."""
    kwargs['region_name'] = 'us-west-2'
    return BedrockClaudeClient(
        model_id="us.anthropic.claude-opus-4-20250514-v1:0",
        **kwargs
    )


# Test function for validation
async def test_bedrock_client():
    """Test function to validate Bedrock client setup."""
    print("Testing Bedrock Claude clients...")
    print(f"Using region: {os.getenv('AWS_REGION', 'eu-central-1')} (Claude 3.x models)")
    print("Claude 4 models will automatically use US regions (us-east-1/us-west-2)")
    print()
    
    # Test Claude 3.5 Sonnet
    try:
        client_35 = create_claude_35_sonnet_client()
        response = await client_35.query("Hello, how are you?")
        print(f"✅ Claude 3.5 Sonnet: {response[:100]}...")
    except Exception as e:
        print(f"❌ Claude 3.5 Sonnet failed: {e}")
    
    # Test Claude 3.7 Sonnet (requires ARN)
    inference_profile_arn = os.getenv("CLAUDE_37_INFERENCE_PROFILE_ARN")
    if inference_profile_arn:
        try:
            client_37 = create_claude_37_sonnet_client(inference_profile_arn)
            response = await client_37.query("Hello, how are you?")
            print(f"✅ Claude 3.7 Sonnet: {response[:100]}...")
        except Exception as e:
            print(f"❌ Claude 3.7 Sonnet failed: {e}")
    else:
        print("⚠️ Claude 3.7 Sonnet skipped (no inference profile ARN)")
    
    # Test Claude 3 Haiku
    try:
        client_haiku = create_claude_haiku_client()
        response = await client_haiku.query("Hello, how are you?")
        print(f"✅ Claude 3 Haiku: {response[:100]}...")
    except Exception as e:
        print(f"❌ Claude 3 Haiku failed: {e}")
    
    # Test Claude 4 Sonnet (US region required)
    try:
        client_4_sonnet = create_claude_4_sonnet_client()
        response = await client_4_sonnet.query("Hello, how are you?")
        print(f"✅ Claude 4 Sonnet (US region): {response[:100]}...")
    except Exception as e:
        print(f"❌ Claude 4 Sonnet (US region) failed: {e}")
        
        # Try alternative US region (us-west-2)
        try:
            print("   🔄 Trying us-west-2 region...")
            client_4_sonnet_west = create_claude_4_sonnet_client_us_west()
            response = await client_4_sonnet_west.query("Hello, how are you?")
            print(f"✅ Claude 4 Sonnet (US West): {response[:100]}...")
        except Exception as e2:
            print(f"❌ Claude 4 Sonnet (US West) also failed: {e2}")
    
    # Test Claude 4 Opus (US region required)
    try:
        client_4_opus = create_claude_4_opus_client()
        response = await client_4_opus.query("Hello, how are you?")
        print(f"✅ Claude 4 Opus (US region): {response[:100]}...")
    except Exception as e:
        print(f"❌ Claude 4 Opus (US region) failed: {e}")
        
        # Try alternative US region (us-west-2)
        try:
            print("   🔄 Trying us-west-2 region...")
            client_4_opus_west = create_claude_4_opus_client_us_west()
            response = await client_4_opus_west.query("Hello, how are you?")
            print(f"✅ Claude 4 Opus (US West): {response[:100]}...")
        except Exception as e2:
            print(f"❌ Claude 4 Opus (US West) also failed: {e2}")
    
    print("\n💡 Claude 4 Regional Requirements (Per AWS Documentation):")
    print("   📍 Source Regions: us-east-1, us-east-2, us-west-2 ONLY")
    print("   📍 Destination Regions: us-east-1, us-east-2, us-west-2")
    print("   ❌ EU/APAC: Claude 4 models NOT available from any EU or APAC source regions")
    print("   ✅ Solution: Code automatically uses US regions for Claude 4 models")
    print(f"   🏠 Your data stays secure - only API calls route through US regions")
    print("\n   Inference Profile IDs used:")
    print("   • Claude 4 Sonnet: us.anthropic.claude-sonnet-4-20250514-v1:0")
    print("   • Claude 4 Opus: us.anthropic.claude-opus-4-20250514-v1:0")


async def check_available_models():
    """Check what Claude models are available in your regions."""
    print("🔍 Commands to check Claude model availability:")
    print()
    print("# Check eu-central-1 (your current region - Claude 3.x models)")
    print("aws bedrock list-foundation-models --region eu-central-1 --by-provider anthropic")
    print("aws bedrock list-inference-profiles --region eu-central-1")
    print()
    print("# Check us-east-1 (required for Claude 4 access)")
    print("aws bedrock list-foundation-models --region us-east-1 --by-provider anthropic")
    print("aws bedrock list-inference-profiles --region us-east-1")
    print()
    print("📋 Expected Results:")
    print("   eu-central-1: Claude 3.x models available")
    print("   us-east-1: Claude 3.x + Claude 4 models available")
    print()
    print("🚨 Important: Claude 4 inference profiles can ONLY be called from:")
    print("   • us-east-1 (N. Virginia)")
    print("   • us-east-2 (Ohio)") 
    print("   • us-west-2 (Oregon)")
    print()
    print("💡 If Claude 4 models still fail:")
    print("   1. Check AWS Bedrock console > Model Access")
    print("   2. Request access to Claude 4 models if needed")
    print("   3. Ensure your AWS credentials work in US regions")
    print("   4. Check your account has permissions for cross-region calls")


if __name__ == "__main__":
    # Run test
    asyncio.run(test_bedrock_client())
    
    # Uncomment to see CLI commands for checking available models
    # asyncio.run(check_available_models())
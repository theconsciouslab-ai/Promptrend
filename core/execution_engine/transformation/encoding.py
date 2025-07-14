"""
Encoding utilities for prompt transformation.
"""

import base64
import urllib.parse
import codecs
import random
from typing import List, Optional


def encode_prompt(prompt: str, method: str) -> str:
    """
    Encode prompt using the specified encoding method.
    
    Args:
        prompt: The original prompt string
        method: Encoding method - one of "base64", "rot13", "url", "hex"
    
    Returns:
        Encoded prompt as string
        
    Raises:
        ValueError: If encoding method is not supported
    """
    encoders = {
        "base64": lambda p: base64.b64encode(p.encode()).decode(),
        "rot13": lambda p: codecs.encode(p, 'rot_13'),
        "url": lambda p: urllib.parse.quote(p),
        "hex": lambda p: p.encode().hex()
    }
    
    if method not in encoders:
        raise ValueError(f"Unsupported encoding method: {method}")
    
    return encoders[method](prompt)


def decode_prompt(encoded: str, method: str) -> str:
    """
    Decode an encoded prompt.
    
    Args:
        encoded: The encoded prompt string
        method: Encoding method used
    
    Returns:
        Decoded prompt as string
        
    Raises:
        ValueError: If encoding method is not supported
    """
    decoders = {
        "base64": lambda e: base64.b64decode(e.encode()).decode(),
        "rot13": lambda e: codecs.decode(e, 'rot_13'),
        "url": lambda e: urllib.parse.unquote(e),
        "hex": lambda e: bytes.fromhex(e).decode()
    }
    
    if method not in decoders:
        raise ValueError(f"Unsupported encoding method: {method}")
    
    return decoders[method](encoded)


def random_encode(prompt: str, methods: Optional[List[str]] = None) -> tuple[str, str]:
    """
    Randomly select an encoding method and apply it.
    
    Args:
        prompt: The original prompt
        methods: Optional list of encoding methods to choose from
    
    Returns:
        Tuple of (encoded_prompt, method_used)
    """
    available_methods = methods or ["base64", "rot13", "url", "hex"]
    method = random.choice(available_methods)
    return encode_prompt(prompt, method), method


# Available encoding methods
ENCODING_METHODS = ["base64", "rot13", "url", "hex"]
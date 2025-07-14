# utils/privacy_tools.py
import logging
import re
import hashlib
import data_collection_config

logger = logging.getLogger("PrivacyTools")

class PrivacyTools:
    """
    Implements privacy protection measures for collected data.
    
    Provides a suite of anonymization and sanitization techniques
    to ensure ethical data handling while maintaining analytical value.
    """
    
    def __init__(self):
        """Initialize privacy tools with configuration settings."""
        # Salt for consistent hashing
        self.hash_salt = data_collection_config.PRIVACY_CONFIG["hash_salt"]
        
        # PII detection patterns
        self.pii_patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
            "phone": r'\b(\+\d{1,3})?\s*\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
            "ssn": r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b',
            "ip_address": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            "credit_card": r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
        }
    
    def anonymize_username(self, username):
        """
        Anonymize a username with consistent hashing.
        
        Args:
            username: Username to anonymize
            
        Returns:
            str: Anonymized username
        """
        if not username or username == "[deleted]":
            return "[deleted]"
            
        # Create consistent hash with salt
        salted_username = (username + self.hash_salt).encode('utf-8')
        hashed = hashlib.sha256(salted_username).hexdigest()
        
        # Use first 8 characters of hash for shorter identifier
        return "user_" + hashed[:8]
    
    def sanitize_text(self, text):
        """
        Sanitize text by removing personally identifiable information.
        
        Args:
            text: Text to sanitize
            
        Returns:
            str: Sanitized text
        """
        if not text:
            return text
            
        sanitized = text
        
        # Apply each PII pattern
        for pii_type, pattern in self.pii_patterns.items():
            # Replace sensitive information with redacted marker
            sanitized = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", sanitized)
        
        return sanitized
    
    def filter_content(self, content, content_type):
        """
        Filter content to remove specific sensitive information.
        
        Args:
            content: Content to filter
            content_type: Type of content for specific filtering rules
            
        Returns:
            Content with sensitive information filtered
        """
        if content_type == "text":
            return self.sanitize_text(content)
        elif content_type == "user":
            return self.anonymize_username(content)
        elif content_type == "json":
            # Recursively sanitize JSON object
            return self._sanitize_json(content)
        else:
            # For unknown types, return as is
            return content
    
    def _sanitize_json(self, json_data):
        """
        Sanitize a JSON object recursively.
        
        Args:
            json_data: JSON object
            
        Returns:
            dict: Sanitized JSON object
        """
        if not isinstance(json_data, dict):
            return json_data
            
        sanitized = {}
        
        for key, value in json_data.items():
            # Check if key indicates sensitive content
            if any(k in key.lower() for k in ["email", "phone", "address", "user", "name", "author"]):
                if isinstance(value, str):
                    sanitized[key] = self.filter_content(value, "text")
                elif isinstance(value, dict):
                    sanitized[key] = self._sanitize_json(value)
                elif isinstance(value, list):
                    sanitized[key] = [self._sanitize_json(item) if isinstance(item, dict) else item for item in value]
                else:
                    sanitized[key] = value
            else:
                # Process value based on type
                if isinstance(value, str):
                    sanitized[key] = self.filter_content(value, "text")
                elif isinstance(value, dict):
                    sanitized[key] = self._sanitize_json(value)
                elif isinstance(value, list):
                    sanitized[key] = [self._sanitize_json(item) if isinstance(item, dict) else item for item in value]
                else:
                    sanitized[key] = value
        
        return sanitized
    
    def anonymize_item(self, item):
        """
        Apply comprehensive anonymization to an item.
        
        Args:
            item: Item to anonymize
            
        Returns:
            dict: Anonymized item
        """
        if not isinstance(item, dict):
            return item
            
        anonymized = {}
        
        # Process each field
        for key, value in item.items():
            if key == "author" or key.endswith("_author"):
                # Anonymize username fields
                anonymized[key] = self.anonymize_username(value)
            elif key == "text" or key == "title" or key == "body" or key == "content":
                # Sanitize text content
                anonymized[key] = self.sanitize_text(value)
            elif key == "comments" and isinstance(value, list):
                # Process comments recursively
                anonymized[key] = [self.anonymize_item(comment) for comment in value]
            elif isinstance(value, dict):
                # Process nested dictionaries
                anonymized[key] = self.anonymize_item(value)
            elif isinstance(value, list):
                # Process lists
                if key in ["users", "authors", "participants"]:
                    # Lists of usernames
                    anonymized[key] = [self.anonymize_username(username) for username in value]
                else:
                    # General lists, process dictionaries inside
                    anonymized[key] = [
                        self.anonymize_item(item) if isinstance(item, dict) else item 
                        for item in value
                    ]
            else:
                # Copy other values as is
                anonymized[key] = value
        
        return anonymized
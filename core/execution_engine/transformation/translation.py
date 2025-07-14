"""
Translation module for prompt transformation pipeline.
Provides free translation capabilities using multiple providers.
"""

import logging
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache

try:
    from deep_translator import GoogleTranslator, MyMemoryTranslator
    from langdetect import detect, LangDetectException
    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False
    logging.warning("Translation libraries not installed. Run: pip install deep-translator langdetect")

logger = logging.getLogger(__name__)


class TranslationProvider(Enum):
    """Available translation providers."""
    GOOGLE = "google"
    MYMEMORY = "mymemory"


@dataclass
class TranslationConfig:
    """Configuration for translation module."""
    target_languages: List[str] = field(default_factory=lambda: [
        "es", "fr", "de", "it", "pt", "ru", "ja", "zh-CN", "ar", "hi"
    ])
    preferred_providers: List[TranslationProvider] = field(default_factory=lambda: [
        TranslationProvider.GOOGLE,
        TranslationProvider.MYMEMORY,
    ])
    preserve_special_tokens: bool = True
    chunk_size: int = 500
    fallback_on_error: bool = True
    detect_source_language: bool = True
    

class TranslationEngine:
    """Translation engine for prompt transformation."""
    
    def __init__(self, config: Optional[TranslationConfig] = None):
        """Initialize the translation engine."""
        self.config = config or TranslationConfig()
        
        if not TRANSLATION_AVAILABLE:
            raise ImportError(
                "Translation libraries not available. "
                "Install with: pip install deep-translator langdetect"
            )
        
        self.special_tokens_pattern = re.compile(
            r'(\[MASK\]|\[REDACTED\]|\###|<<<|>>>|\*\*\*|---|___|\[.*?\])'
        )
        
        logger.info(f"Translation engine initialized with {len(self.config.target_languages)} languages")
    
    @lru_cache(maxsize=128)
    def translate(
        self, 
        text: str, 
        target_language: str,
        source_language: Optional[str] = None
    ) -> Tuple[str, Dict[str, any]]:
        """
        Translate text to target language with caching.
        
        Args:
            text: Text to translate
            target_language: Target language code
            source_language: Source language code (auto-detected if None)
            
        Returns:
            Tuple of (translated_text, metadata)
        """
        if not text or not text.strip():
            return text, {"error": "Empty text", "success": False}
        
        # Detect source language if needed
        if source_language is None and self.config.detect_source_language:
            try:
                source_language = detect(text)
            except LangDetectException:
                source_language = "auto"
        
        # Preserve special tokens if needed
        token_map = {}
        if self.config.preserve_special_tokens:
            text, token_map = self._extract_special_tokens(text)
        
        # Perform translation
        translated_text = None
        metadata = {
            "source_language": source_language,
            "target_language": target_language,
            "provider": None,
            "success": False
        }
        
        for provider in self.config.preferred_providers:
            try:
                translated_text = self._translate_with_provider(
                    text, source_language, target_language, provider
                )
                
                if translated_text and translated_text != text:
                    metadata["provider"] = provider.value
                    metadata["success"] = True
                    break
                    
            except Exception as e:
                logger.debug(f"Translation failed with {provider.value}: {e}")
                if not self.config.fallback_on_error:
                    raise
        
        # Restore special tokens
        if translated_text and self.config.preserve_special_tokens and token_map:
            translated_text = self._restore_special_tokens(translated_text, token_map)
        
        # Fallback to original if translation failed
        if not translated_text:
            translated_text = text
            metadata["error"] = "All providers failed"
        
        return translated_text, metadata
    
    def _translate_with_provider(
        self, 
        text: str, 
        source_lang: str, 
        target_lang: str, 
        provider: TranslationProvider
    ) -> str:
        """Translate using specific provider."""
        if provider == TranslationProvider.GOOGLE:
            translator = GoogleTranslator(source=source_lang, target=target_lang)
            return translator.translate(text)
            
        elif provider == TranslationProvider.MYMEMORY:
            # MyMemory has character limit
            if len(text) > 1000:
                chunks = self._split_text(text, 900)
                return " ".join(
                    MyMemoryTranslator(source=source_lang, target=target_lang).translate(chunk)
                    for chunk in chunks
                )
            else:
                translator = MyMemoryTranslator(source=source_lang, target=target_lang)
                return translator.translate(text)
        
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _split_text(self, text: str, max_length: int) -> List[str]:
        """Split text into chunks for translation."""
        # Try to split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= max_length:
                current_chunk += (" " + sentence if current_chunk else sentence)
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _extract_special_tokens(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Extract and replace special tokens with placeholders."""
        token_map = {}
        counter = 0
        
        def replace_token(match):
            nonlocal counter
            token = match.group(0)
            placeholder = f"__TOKEN_{counter}__"
            token_map[placeholder] = token
            counter += 1
            return placeholder
        
        cleaned_text = self.special_tokens_pattern.sub(replace_token, text)
        return cleaned_text, token_map
    
    def _restore_special_tokens(self, text: str, token_map: Dict[str, str]) -> str:
        """Restore special tokens from placeholders."""
        for placeholder, token in token_map.items():
            text = text.replace(placeholder, token)
        return text
    
    def get_supported_languages(self) -> List[Tuple[str, str]]:
        """Get list of supported languages with their names."""
        language_names = {
            "es": "Spanish", "fr": "French", "de": "German",
            "it": "Italian", "pt": "Portuguese", "ru": "Russian",
            "ja": "Japanese", "zh-CN": "Chinese (Simplified)",
            "ar": "Arabic", "hi": "Hindi", "ko": "Korean",
            "pl": "Polish", "tr": "Turkish", "sv": "Swedish",
            "nl": "Dutch"
        }
        return [(code, language_names.get(code, code.upper())) 
                for code in self.config.target_languages]


# Convenience function
def translate_prompt(
    prompt: str, 
    target_language: str,
    preserve_tokens: bool = True
) -> Tuple[str, Dict]:
    """Convenience function to translate a single prompt."""
    config = TranslationConfig(preserve_special_tokens=preserve_tokens)
    engine = TranslationEngine(config)
    return engine.translate(prompt, target_language)
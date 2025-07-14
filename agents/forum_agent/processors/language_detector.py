# processors/language_detector.py
import logging
from typing import Dict, Any, List, Optional
import aiohttp
from bs4 import BeautifulSoup
from langdetect import detect, DetectorFactory
import iso639

from agents.forum_agent.forum_config import (
        HEADERS,
        GOOGLE_TRANSLATE_API_KEY,
        TRANSLATION_API_TYPE,
        DEEPL_API_KEY,
)

logger = logging.getLogger("LanguageDetector")

# Ensure consistent language detection results
DetectorFactory.seed = 0

class LanguageDetector:
    """
    Detects languages and provides translation services.
    
    This component identifies the primary language of forums and threads,
    and conditionally translates content for analysis when needed.
    """
    
    def __init__(self):
        """Initialize the language detector."""
        # No model loading needed for langdetect
        logger.info("Initialized langdetect for language detection")
        
        # Cache for forum languages
        self.forum_language_cache = {}
    
    async def detect_forum_language(self, url: str) -> str:
        """
        Detect the primary language of a forum.
        
        Args:
            url: Forum URL
            
        Returns:
            str: ISO 639-1 language code (e.g., 'en', 'ar', 'zh')
        """
        # Check cache first
        if url in self.forum_language_cache:
            return self.forum_language_cache[url]
            
        try:
            # Fetch forum homepage
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=HEADERS) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to access {url}: Status {response.status}")
                        return 'en'  # Default to English
                        
                    html = await response.text()
                    
            # Extract text for language detection
            sample_text = self._extract_text_for_detection(html)
            
            # Detect language
            language_code = self._detect_language(sample_text)
            
            # Update cache
            self.forum_language_cache[url] = language_code
            
            return language_code
                    
        except Exception as e:
            logger.error(f"Error detecting forum language for {url}: {str(e)}")
            return 'en'  # Default to English
    
    def detect_text_language(self, text: str) -> str:
        """
        Detect the language of a text snippet.
        
        Args:
            text: Text to analyze
            
        Returns:
            str: ISO 639-1 language code
        """
        if not text or len(text) < 20:
            return 'en'  # Not enough text to reliably detect
            
        return self._detect_language(text)
    
    async def translate_text(self, text: str, source_lang: str, target_lang: str = 'en') -> str:
        """
        Translate text from source language to target language.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code (default: English)
            
        Returns:
            str: Translated text
        """
        # Skip translation if source and target are the same
        if source_lang == target_lang:
            return text
            
        try:
            # Use translation API (Google Translate API)
            if TRANSLATION_API_TYPE == 'google':
                return await self._translate_google(text, source_lang, target_lang)
            elif TRANSLATION_API_TYPE == 'deepl':
                return await self._translate_deepl(text, source_lang, target_lang)
            else:
                logger.warning(f"Unsupported translation API: {TRANSLATION_API_TYPE}")
                return text
                
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return text  # Return original text if translation fails
    
    def _extract_text_for_detection(self, html: str, max_length: int = 1000) -> str:
        """
        Extract representative text from HTML for language detection.
        
        Args:
            html: HTML content
            max_length: Maximum text length to extract
            
        Returns:
            str: Extracted text for language detection
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script, style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Get text from main content areas
            content_selectors = [
                "main", "article", "#content", ".content", 
                "#main", ".main", ".post", ".thread"
            ]
            
            for selector in content_selectors:
                content = soup.select(selector)
                if content:
                    text = " ".join([element.get_text() for element in content])
                    if len(text) > 200:  # Reasonable sample
                        return text[:max_length]
            
            # Fallback to body text
            text = soup.body.get_text() if soup.body else soup.get_text()
            text = " ".join(text.split())  # Normalize whitespace
            
            return text[:max_length]
            
        except Exception as e:
            logger.error(f"Error extracting text for detection: {str(e)}")
            return ""
    
    def _detect_language(self, text: str) -> str:
        """
        Detect the language of text using langdetect.
        
        Args:
            text: Text to analyze
            
        Returns:
            str: ISO 639-1 language code
        """
        if not text:
            return 'en'  # Default to English if text empty
            
        try:
            # Detect language using langdetect
            lang_code = detect(text)
            
            # Ensure ISO 639-1 format (langdetect already returns ISO 639-1 codes)
            if len(lang_code) > 2:
                try:
                    lang_code = iso639.to_iso639_1(lang_code)
                except:
                    # If conversion fails, return first 2 characters or default to English
                    lang_code = lang_code[:2] if len(lang_code) >= 2 else 'en'
            
            return lang_code
            
        except Exception as e:
            logger.error(f"Language detection error: {str(e)}")
            return 'en'  # Default to English
    
    async def _translate_google(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate text using Google Translate API.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            str: Translated text
        """
        try:
            url = "https://translation.googleapis.com/language/translate/v2"
            params = {
                "q": text,
                "source": source_lang,
                "target": target_lang,
                "format": "text",
                "key": GOOGLE_TRANSLATE_API_KEY
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, params=params) as response:
                    if response.status != 200:
                        logger.warning(f"Translation failed: Status {response.status}")
                        return text
                        
                    data = await response.json()
                    
                    if "data" in data and "translations" in data["data"] and data["data"]["translations"]:
                        return data["data"]["translations"][0]["translatedText"]
                    else:
                        logger.warning("Unexpected translation API response format")
                        return text
                        
        except Exception as e:
            logger.error(f"Google Translate API error: {str(e)}")
            return text
    
    async def _translate_deepl(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate text using DeepL API.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            str: Translated text
        """
        try:
            url = "https://api-free.deepl.com/v2/translate"
            headers = {
                "Authorization": f"DeepL-Auth-Key {DEEPL_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # Map ISO 639-1 codes to DeepL format if needed
            deepl_source = source_lang.upper() if len(source_lang) == 2 else source_lang
            deepl_target = target_lang.upper() if len(target_lang) == 2 else target_lang
            
            payload = {
                "text": [text],
                "source_lang": deepl_source,
                "target_lang": deepl_target
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        logger.warning(f"DeepL translation failed: Status {response.status}")
                        return text
                        
                    data = await response.json()
                    
                    if "translations" in data and data["translations"]:
                        return data["translations"][0]["text"]
                    else:
                        logger.warning("Unexpected DeepL API response format")
                        return text
                        
        except Exception as e:
            logger.error(f"DeepL API error: {str(e)}")
            return text
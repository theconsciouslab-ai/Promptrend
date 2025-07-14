# processors/artifact_extractor.py

import re
import logging
import os
import aiohttp
from urllib.parse import urlparse
from agents.discord_agent.discord_config import (
    MIN_INLINE_CODE_LENGTH,
    RELEVANT_URL_DOMAINS,
    CODE_FILE_EXTENSIONS,
    TEXT_FILE_EXTENSIONS,
)
logger = logging.getLogger("ArtifactExtractor")

class ArtifactExtractor:
    """
    Extracts code snippets, links, and other artifacts from Discord messages.
    
    These artifacts are used for more detailed LLM analysis, especially
    for detecting vulnerability demonstrations or proof-of-concept code.
    """
    def __init__(self):
        self.code_block_pattern = re.compile(r'```(?:\w+)?\n([\s\S]*?)\n```')
        self.inline_code_pattern = re.compile(r'`([^`]+)`')
        self.url_pattern = re.compile(r'https?://\S+')
        
        self.code_file_extensions = CODE_FILE_EXTENSIONS
        self.text_file_extensions = TEXT_FILE_EXTENSIONS
        
        # Add max file size limit
        self.max_file_size = 1024 * 1024  # 1MB limit

    async def extract_artifacts(self, messages):
        """
        Extract artifacts from a list of messages.
        
        Args:
            messages: List of Discord message objects
            
        Returns:
            dict: Dictionary of extracted artifacts (code, links, files)
        """
        artifacts = {
            'code': [],
            'links': [],
            'files': [],
            'text_files': []
        }

        for message in messages:
            # Extract code blocks from message content
            if message.content:
                code_blocks = self.code_block_pattern.findall(message.content)
                if code_blocks:
                    logger.debug(f"Found {len(code_blocks)} code blocks in message")
                    artifacts['code'].extend(code_blocks)

                # Extract longer inline code snippets
                inline_codes = self.inline_code_pattern.findall(message.content)
                for code in inline_codes:
                    if len(code) > MIN_INLINE_CODE_LENGTH:
                        artifacts['code'].append(code)

                # Extract URLs
                urls = self.url_pattern.findall(message.content)
                relevant_urls = [
                    url for url in urls if any(site in urlparse(url).netloc for site in RELEVANT_URL_DOMAINS)
                ]
                
                if relevant_urls:
                    logger.debug(f"Found {len(relevant_urls)} relevant URLs in message")
                    artifacts['links'].extend(relevant_urls)

            # Process attachments
            for attachment in message.attachments:
                filename = attachment.filename.lower()
                _, ext = os.path.splitext(filename)
                
                logger.info(f"Processing attachment: {filename} ({attachment.size} bytes) - Extension: {ext}")

                # Check file size limit
                if attachment.size > self.max_file_size:
                    logger.warning(f"File too large, skipping: {filename} ({attachment.size} bytes)")
                    continue

                # Create a basic file info dictionary
                file_info = {
                    'url': attachment.url,
                    'filename': attachment.filename,
                    'size': attachment.size,
                    'extension': ext
                }

                # Handle text files (.txt, .md, .csv, .log)
                if ext in self.text_file_extensions:
                    logger.info(f"Found text file: {filename} - attempting download")
                    try:
                        text_content = await self._download_text_attachment(attachment.url)
                        if text_content:
                            logger.info(f"✅ Successfully downloaded {filename}: {len(text_content)} chars")
                            artifacts['text_files'].append(text_content)
                            file_info['downloaded'] = True
                            file_info['content_length'] = len(text_content)
                        else:
                            logger.warning(f"❌ Failed to download content from: {filename}")
                            file_info['downloaded'] = False
                        artifacts['files'].append(file_info)
                    except Exception as e:
                        logger.error(f"❌ Error downloading text file {filename}: {e}")
                        file_info['downloaded'] = False
                        file_info['error'] = str(e)
                        artifacts['files'].append(file_info)
                
                # Handle code files
                elif ext in self.code_file_extensions:
                    logger.info(f"Found code file: {filename}")
                    artifacts['files'].append(file_info)
                    
                    # Also try to download code file content for analysis
                    try:
                        code_content = await self._download_text_attachment(attachment.url)
                        if code_content:
                            logger.info(f"✅ Successfully downloaded code file: {filename} - {len(code_content)} chars")
                            artifacts['code'].append(code_content)
                            file_info['downloaded'] = True
                        else:
                            file_info['downloaded'] = False
                    except Exception as e:
                        logger.warning(f"Failed to download code file content: {filename} - {e}")
                        file_info['downloaded'] = False
                
                # Handle all other file types
                else:
                    logger.info(f"Unsupported file type: {filename} - {ext}")
                    artifacts['files'].append(file_info)
       
        # Log summary
        logger.info(f"📊 Extraction Summary:")
        for key, items in artifacts.items():
            if items:
                logger.info(f"  - {key}: {len(items)} items")

        # Only keep non-empty categories in the result
        return {k: v for k, v in artifacts.items() if v}

    async def _download_text_attachment(self, url):
        """
        Downloads and returns the content of a text file from a URL.
        
        Args:
            url: URL to the text file
            
        Returns:
            str: Content of the text file, or None if download failed
        """
        try:
            timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                logger.debug(f"Attempting to download: {url}")
                async with session.get(url) as resp:
                    if resp.status == 200:
                        # Check content length
                        content_length = resp.headers.get('content-length')
                        if content_length and int(content_length) > self.max_file_size:
                            logger.warning(f"Content too large: {content_length} bytes")
                            return None
                            
                        content = await resp.text(encoding='utf-8', errors='ignore')
                        logger.debug(f"✅ Downloaded {len(content)} characters from {url}")
                        return content
                    else:
                        logger.warning(f"❌ HTTP error downloading {url}: {resp.status}")
                        return None
        except aiohttp.ClientError as e:
            logger.error(f"❌ Network error downloading {url}: {e}")
        except Exception as e:
            logger.error(f"❌ Unexpected error downloading {url}: {e}")
        return None
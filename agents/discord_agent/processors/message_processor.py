# processors/message_processor.py
import re
import logging
import discord
from agents.discord_agent.processors.artifact_extractor import ArtifactExtractor
from agents.discord_agent.processors.llm_analyzer import LLMAnalyzer

from agents.discord_agent.discord_config import (
    DISCUSSION_SCORE_WEIGHT,
    CODE_SCORE_WEIGHT,
    VULNERABILITY_KEYWORDS,
    RELEVANCE_THRESHOLD,
)

logger = logging.getLogger("MessageProcessor")

class MessageProcessor:
    """
    Enhanced message processor that relies solely on LLM analysis for vulnerability detection.
    
    Removes prompt extraction dependency and focuses on comprehensive LLM-based 
    vulnerability analysis for more accurate detection.
    """
    
    def __init__(self):
        """Initialize the message processor."""
        self.artifact_extractor = ArtifactExtractor()
        self.llm_analyzer = LLMAnalyzer()
        self.relevance_threshold = RELEVANCE_THRESHOLD
        self.keyword_lexicon = set(VULNERABILITY_KEYWORDS)

    async def process_message_group(self, messages, channel):
        """
        Process a group of messages as a conversation using enhanced LLM analysis.
        """
        if not messages:
            return None

        logger.info(f"🔍 Processing {len(messages)} messages from #{channel.name}")
        
        # Check for attachments first
        attachment_count = sum(len(msg.attachments) for msg in messages)
        if attachment_count > 0:
            logger.info(f"📎 Found {attachment_count} attachments to process")
            for msg in messages:
                for att in msg.attachments:
                    logger.info(f"  - {att.filename} ({att.size} bytes)")

        # ✅ Step 1: Extract artifacts
        logger.info("📥 Extracting artifacts...")
        artifacts = await self.artifact_extractor.extract_artifacts(messages)
        
        # Debug artifact extraction results
        if artifacts:
            logger.info(f"📦 Artifacts extracted:")
            for key, items in artifacts.items():
                logger.info(f"  - {key}: {len(items)} items")
                if key == 'text_files':
                    for i, content in enumerate(items):
                        logger.info(f"    Text file {i+1}: {len(content)} characters")
        else:
            logger.info("📦 No artifacts extracted")

        # ✅ Step 2: Extract text content including .txt files
        text_content = self._extract_text_content(messages, artifacts)
        logger.info(f"📝 Total text content: {len(text_content)} characters")

        # ✅ Step 3: Quick keyword relevance check (optional pre-filter)
        if not self._keyword_relevance_check(text_content):
            logger.debug("Message failed keyword relevance check")
            return None

        # ✅ Step 4: Build comprehensive LLM context
        context = self._prepare_analysis_context(messages, artifacts)

        # ✅ Step 5: Comprehensive LLM analysis
        logger.info("🤖 Starting comprehensive LLM analysis...")
        
        try:
            # Get comprehensive analysis from enhanced LLM analyzer
            comprehensive_analysis = await self.llm_analyzer.analyze_content_comprehensive(context, artifacts)
            
            # Extract scores for compatibility with existing system
            overall_score = comprehensive_analysis.get("overall_score", 0.0)
            confidence = comprehensive_analysis.get("confidence", 0.0)
            
            # Also get individual scores for detailed tracking
            discussion_score = await self.llm_analyzer.analyze_discussion(context)
            
            # Process code snippets including text from files
            code_snippets = artifacts.get('code', [])
            if 'text_files' in artifacts:
                code_snippets.extend(artifacts['text_files'])
            
            code_score = await self.llm_analyzer.analyze_code(code_snippets)
            
            # Calculate final score using comprehensive analysis as primary
            final_score = overall_score
            
            # Log analysis results
            logger.info(f"📊 Analysis Results:")
            logger.info(f"  - Overall Score: {overall_score:.3f}")
            logger.info(f"  - Confidence: {confidence:.3f}")
            logger.info(f"  - Discussion Score: {discussion_score:.3f}")
            logger.info(f"  - Code Score: {code_score:.3f}")
            logger.info(f"  - Final Score: {final_score:.3f}")
            logger.info(f"  - Vulnerability Detected: {comprehensive_analysis.get('vulnerability_detected', False)}")
            logger.info(f"  - Type: {comprehensive_analysis.get('vulnerability_type', 'Unknown')}")
            
            # Check if this meets our threshold
            if final_score < self.relevance_threshold:
                logger.debug(f"Message below relevance threshold: {final_score} < {self.relevance_threshold}")
                return None

            # ✅ Step 6: Build comprehensive result
            result = {
                'is_vulnerability': True,
                'channel_id': channel.id,
                'guild_id': channel.guild.id if channel.guild else None,
                'channel_name': channel.name,
                'server_name': channel.guild.name if channel.guild else None,
                'message_ids': [str(m.id) for m in messages],
                'message_url': self._build_message_url(messages[0]),
                'timestamp': max(m.created_at for m in messages).isoformat(),
                'authors': list(set(m.author.name for m in messages)),
                'content': text_content,
                'artifacts': artifacts,
                
                # Enhanced analysis results
                'comprehensive_analysis': comprehensive_analysis,
                
                # Legacy score format for compatibility
                'scores': {
                    'discussion': discussion_score,
                    'code': code_score,
                    'overall': overall_score,
                    'confidence': confidence
                },
                'final_score': final_score,
                'relevance_score': final_score,
                
                # Enhanced vulnerability details
                'vulnerability_type': comprehensive_analysis.get('vulnerability_type', 'Unknown'),
                'sophistication_level': comprehensive_analysis.get('sophistication_level', 'low'),
                'potential_impact': comprehensive_analysis.get('potential_impact', 'low'),
                'key_techniques': comprehensive_analysis.get('key_techniques', []),
                'target_models': comprehensive_analysis.get('target_models', []),
                'extracted_prompts': comprehensive_analysis.get('extracted_prompts', []),
                'potential_mitigations': comprehensive_analysis.get('potential_mitigations', []),
                'analysis_summary': comprehensive_analysis.get('summary', '')
            }
            
            logger.info(f"✅ Detected vulnerability in {channel.name}")
            logger.info(f"   Type: {result['vulnerability_type']}")
            logger.info(f"   Score: {final_score:.3f}")
            logger.info(f"   Techniques: {', '.join(result['key_techniques'][:3])}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error during LLM analysis: {e}")
            return None

    def _extract_text_content(self, messages, artifacts):
        """Extract text content from messages and artifacts."""
        contents = [m.content for m in messages if m.content]

        # Include text file content
        if artifacts.get('text_files'):
            logger.debug(f"Including {len(artifacts['text_files'])} text files in analysis")
            contents.extend(artifacts['text_files'])

        return "\n".join(contents)

    def _keyword_relevance_check(self, text):
        """
        Quick keyword check to filter obviously irrelevant content.
        
        Args:
            text: Message text to check
            
        Returns:
            bool: True if the text might be relevant
        """
        if not text:
            return False
            
        text_lower = text.lower()
        
        # Check for keyword matches
        for keyword in self.keyword_lexicon:
            if keyword.lower() in text_lower:
                logger.debug(f"Keyword match found: {keyword}")
                return True
        
        # Also check for common LLM/AI terms that might not be in the main lexicon
        ai_terms = ['chatgpt', 'gpt', 'claude', 'llm', 'ai model', 'language model', 
                   'prompt', 'jailbreak', 'injection', 'bypass', 'exploit']
        
        for term in ai_terms:
            if term in text_lower:
                logger.debug(f"AI term match found: {term}")
                return True
        
        return False

    def _prepare_analysis_context(self, messages, artifacts):
        """
        Prepare a comprehensive context for LLM analysis.
        
        Args:
            messages: List of message objects
            artifacts: Dictionary of extracted artifacts
            
        Returns:
            str: Formatted context for LLM analysis
        """
        # Sort messages by timestamp
        sorted_msgs = sorted(messages, key=lambda m: m.created_at)
        
        # Create a conversation transcript
        context_parts = ["=== DISCORD CONVERSATION ==="]
        
        for msg in sorted_msgs:
            author = msg.author.name
            content = msg.content or "[No text content]"
            timestamp = msg.created_at.strftime("%Y-%m-%d %H:%M:%S UTC")
            
            # Include attachment info in the transcript
            attachment_info = ""
            if msg.attachments:
                attachment_info = f" [Attachments: {', '.join(att.filename for att in msg.attachments)}]"
            
            context_parts.append(f"[{timestamp}] {author}: {content}{attachment_info}")
        
        context_parts.append("=== END CONVERSATION ===")
        
        # Add artifact sections
        if artifacts.get('code'):
            context_parts.append("\n=== CODE SNIPPETS ===")
            for i, snippet in enumerate(artifacts['code'], 1):
                context_parts.append(f"\n--- Code Snippet {i} ---")
                context_parts.append(snippet)
            context_parts.append("=== END CODE SNIPPETS ===")
        
        if artifacts.get('text_files'):
            context_parts.append("\n=== TEXT FILE CONTENT ===")
            for i, content in enumerate(artifacts['text_files'], 1):
                context_parts.append(f"\n--- Text File {i} ---")
                context_parts.append(content)
            context_parts.append("=== END TEXT FILE CONTENT ===")
            
        if artifacts.get('links'):
            context_parts.append("\n=== RELEVANT LINKS ===")
            for link in artifacts['links']:
                context_parts.append(f"- {link}")
            context_parts.append("=== END LINKS ===")
            
        if artifacts.get('files'):
            context_parts.append("\n=== FILE ATTACHMENTS ===")
            for file_info in artifacts['files']:
                context_parts.append(f"- {file_info['filename']} ({file_info['size']} bytes)")
            context_parts.append("=== END FILE ATTACHMENTS ===")
            
        return "\n".join(context_parts)

    def _build_message_url(self, message):
        """Build a Discord message URL for reference."""
        if not message.guild:
            return None
        return f"https://discord.com/channels/{message.guild.id}/{message.channel.id}/{message.id}"
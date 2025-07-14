import logging
from datetime import datetime, timedelta, timezone
import asyncio
import aiohttp
from typing import Dict, Optional, Any

from agents.forum_agent.forum_classifier import ForumClassifier
from agents.forum_agent.collectors.base_collector import BaseForumCollector
from agents.forum_agent.collectors.vbulletin_collector import VBulletinCollector
from agents.forum_agent.collectors.discourse_collector import DiscourseCollector
from agents.forum_agent.collectors.phpbb_collector import PhpBBCollector
from agents.forum_agent.collectors.custom_collector import CustomCollector
from agents.forum_agent.processors.language_detector import LanguageDetector
from agents.forum_agent.processors.cultural_analyzer import CulturalAnalyzer
from agents.forum_agent.processors.llm_analyzer import LLMAnalyzer
from agents.forum_agent.processors.prompt_pattern_matcher import PromptPatternMatcher
from agents.forum_agent.utils.rate_limiter import RateLimiter
from agents.forum_agent.utils.ip_rotator import IPRotator
from agents.forum_agent.utils.storage_manager import StorageManager
from agents.forum_agent.forum_config import (
    TARGET_FORUMS,
    INTER_FORUM_DELAY,
    FORUM_LOOKBACK_DAYS,
    MULTILINGUAL_LEXICONS,
    CULTURAL_WEIGHTS,
    PROMPT_TEMPLATES,
    ENABLE_TRANSLATION,
    FORUM_DATA_PATH,
    
)
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/forum_agent.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ForumAgent")

class ForumAgent:
    def __init__(self, region_id= None):
        self.forum_classifier = ForumClassifier()
        self.language_detector = LanguageDetector()
        self.cultural_analyzer = CulturalAnalyzer()
        self.llm_analyzer = LLMAnalyzer()
        self.region_id = region_id
        self.prompt_pattern_matcher = PromptPatternMatcher()
        self.rate_limiter = RateLimiter()
        self.ip_rotator = IPRotator()
        self.storage_manager = StorageManager(storage_dir=FORUM_DATA_PATH)

        self.collectors = {
            "vbulletin": VBulletinCollector(self),
            "discourse": DiscourseCollector(self),
            "phpbb": PhpBBCollector(self),
            "custom": CustomCollector(self)
        }

        self.target_forums = TARGET_FORUMS
        self.forum_metadata = {}
        self.last_check_times = {}
        self.stats = {
            "forums_scanned": 0,
            "threads_processed": 0,
            "vulnerabilities_found": 0,
            "languages_encountered": set(),
            "high_scoring_threads": 0,
            "pattern_matches": 0,
            "cultural_hits": 0,
            "technical_hits": 0
        }

    async def run(self):
        logger.info(f"Starting Discussion Forums agent for {len(self.target_forums)} forums")
        try:
            async with aiohttp.ClientSession() as session:
                self.session = session
                await self._load_forum_state()
                for forum_id, forum_config in self.target_forums.items():
                    try:
                        await self._process_forum(forum_id, forum_config)
                    except Exception as e:
                        logger.error(f"Error processing forum {forum_id}: {str(e)}")
                    await asyncio.sleep(INTER_FORUM_DELAY)
                await self._save_forum_state()
                logger.info(f"Discussion Forums agent completed run: {self.stats}")
        except Exception as e:
            logger.error(f"Error in Discussion Forums agent: {str(e)}")

    async def _load_forum_state(self):
        try:
            state = self.storage_manager.load_agent_state()
            if state:
                self.forum_metadata = state.get('forum_metadata', {})
                self.last_check_times = state.get('last_check_times', {})
                logger.info(f"Loaded state for {len(self.forum_metadata)} forums")
        except Exception as e:
            logger.warning(f"Failed to load forum state: {str(e)}")

    async def _save_forum_state(self):
        try:
            state = {
                'forum_metadata': self.forum_metadata,
                'last_check_times': self.last_check_times,
                'stats': {k: v for k, v in self.stats.items() if k != 'languages_encountered'},
                'languages_encountered': list(self.stats['languages_encountered'])
            }
            self.storage_manager.save_agent_state(state)
            logger.info(f"Saved state for {len(self.forum_metadata)} forums")
        except Exception as e:
            logger.warning(f"Failed to save forum state: {str(e)}")

    async def _process_forum(self, forum_id: str, forum_config: Dict[str, Any]):
        logger.info(f"Processing forum: {forum_id} ({forum_config.get('name', 'Unknown')})")
        platform_type = forum_config.get('platform_type', 'custom')
        collector = self.collectors.get(platform_type)
        if not collector:
            logger.error(f"Unsupported platform type: {platform_type}")
            return

        if forum_id in self.forum_metadata and 'language' in self.forum_metadata[forum_id]:
            forum_language = self.forum_metadata[forum_id]['language']
        else:
            forum_language = await self.language_detector.detect_forum_language(forum_config['url'])
            if forum_id not in self.forum_metadata:
                self.forum_metadata[forum_id] = {}
            self.forum_metadata[forum_id]['language'] = forum_language

        self.stats['languages_encountered'].add(forum_language)
        since_time = self._get_since_time(forum_id)
        lexicon = MULTILINGUAL_LEXICONS.get(forum_language, MULTILINGUAL_LEXICONS['en'])

        threads = await collector.collect_threads(forum_config['url'], lexicon, since_time, forum_language)
        logger.info(f"Collected {len(threads)} threads from {forum_id}")

        for i, thread in enumerate(threads[:3]):
            logger.info(f"Example thread {i+1} content preview:\n{thread.get('content', '')[:600]}")

        for thread in threads:
            cultural_weights = CULTURAL_WEIGHTS.get(forum_language, CULTURAL_WEIGHTS['default'])
            result = await self._process_thread(thread, forum_language, cultural_weights)
            if result:
                self.storage_manager.store_vulnerability(result)
                final_score = result.get("scores", {}).get("final", 0.0)
                self.stats["vulnerabilities_found"] += int(result.get('is_vulnerability', False))
                logger.info(f"Stored vulnerability: {result.get('thread_title')} (Score: {final_score:.2f})")

        self.last_check_times[forum_id] = datetime.now(timezone.utc)
        self.stats["forums_scanned"] += 1
        self.stats["threads_processed"] += len(threads)

    def _get_since_time(self, forum_id: str) -> datetime:
        if forum_id in self.last_check_times:
            return self.last_check_times[forum_id]
        stored_time = self.storage_manager.get_last_check_time(forum_id)
        if stored_time:
            self.last_check_times[forum_id] = stored_time
            return stored_time
        return datetime.now(timezone.utc) - timedelta(days=FORUM_LOOKBACK_DAYS)

    async def _process_thread(self, thread: Dict[str, Any], language: str, cultural_weights: Dict[str, float]) -> Optional[Dict[str, Any]]:
        content = thread.get('content')
        if not content:
            logger.warning(f"Thread {thread.get('thread_id')} has empty content. Attempting to reconstruct from posts.")
            posts = thread.get('posts', [])
            content = "\n\n".join(p.get('content', '') for p in posts if p.get('content'))
            thread['content'] = content
            if not content:
                logger.warning(f"Thread {thread.get('thread_id')} still empty after fallback. Skipping.")
                return None

        try:
            if language != 'en' and ENABLE_TRANSLATION:
                translated_content = await self.language_detector.translate_text(thread['content'], language, 'en')
                thread['translated_content'] = translated_content

            pattern_score, pattern_matches = self.prompt_pattern_matcher.analyze_content(thread)
            cultural_score = await self.cultural_analyzer.analyze_cultural_context(thread, language)
            technical_score = await self.cultural_analyzer.analyze_technical_content(thread)
            prompt = self._construct_cultural_context_prompt(thread, language)
            llm_score =  self.llm_analyzer.analyze(prompt)

                        # Compute final score
            final_score = (
                pattern_score * 0.3 +
                cultural_score * cultural_weights['cultural'] +
                technical_score * cultural_weights['technical'] +
                llm_score * cultural_weights['llm']
            )

            logger.info(
                f"[{thread.get('title', '')[:60]}] Score breakdown → "
                f"Pattern: {pattern_score:.2f}, Cultural: {cultural_score:.2f}, "
                f"Technical: {technical_score:.2f}, LLM: {llm_score:.2f}, Final: {final_score:.2f}"
            )

            # 🔢 Increment metrics
            if final_score >= 0.5:
                self.stats["high_scoring_threads"] += 1
            if pattern_score >= 0.2:
                self.stats["pattern_matches"] += 1
            if cultural_score >= 0.5:
                self.stats["cultural_hits"] += 1
            if technical_score >= 0.3:
                self.stats["technical_hits"] += 1

            return {
                'is_vulnerability': final_score >= 0.2,
                'forum_id': thread.get('forum_id', 'unknown'),
                'forum_name': thread.get('forum_name', 'Unknown'),
                'thread_id': thread.get('thread_id', 'unknown'),
                'thread_title': thread.get('title', 'Untitled'),
                'thread_url': thread.get('url'),
                'author': thread.get('author'),
                'date': thread.get('date'),
                'language': language,
                'translated': language != 'en',
                'content_snippet': thread.get('translated_content', thread.get('content', ''))[:300],
                'pattern_matches': pattern_matches,
                'extracted_prompts': self.prompt_pattern_matcher.extract_prompt_candidates(thread),
                'scores': {
                    'pattern': pattern_score,
                    'cultural': cultural_score,
                    'technical': technical_score,
                    'llm': llm_score,
                    'final': final_score
                },
                'timestamp': datetime.now(timezone.utc).isoformat()
            }


        except Exception as e:
            logger.error(f"Error processing thread: {str(e)}")
            return None

    def _construct_cultural_context_prompt(self, thread: Dict[str, Any], language: str) -> str:
        template = PROMPT_TEMPLATES.get(language, PROMPT_TEMPLATES['default'])
        content = thread.get('translated_content', thread.get('content', ''))
        return template.format(
            title=thread.get('title', 'Untitled'),
            content=content,
            language=language,
            forum_name=thread.get('forum_name', 'Unknown Forum')
        )

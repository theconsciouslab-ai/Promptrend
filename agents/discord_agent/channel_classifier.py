# channel_classifier.py
import asyncio
import discord
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from agents.discord_agent.discord_config import (  
SECURITY_KEYWORDS,
TECHNICAL_KEYWORDS,
MODEL_KEYWORDS,
LDA_NUM_TOPICS,
LDA_MAX_DF,
LDA_MIN_DF,
LDA_TOPIC_SCORE_THRESHOLD,
CATEGORY_CONFIDENCE,

)

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

logger = logging.getLogger("ChannelClassifier")

class ChannelClassifier:
    """
    Classifies Discord channels based on topic relevance to LLM vulnerabilities.
    
    Modified to monitor ALL channels while still categorizing them for analysis.
    """
    
    def __init__(self):
        """Initialize the channel classifier."""
        self.stop_words = set(stopwords.words('english'))
        self.channel_metadata = {}
        self.categories = {
            "security": set(SECURITY_KEYWORDS),
            "technical": set(TECHNICAL_KEYWORDS),
            "model_specific": set(MODEL_KEYWORDS),
            "general": set()
        }

        self.n_topics = LDA_NUM_TOPICS
        self.vectorizer = TfidfVectorizer(
            max_df=LDA_MAX_DF,
            min_df=LDA_MIN_DF,
            stop_words='english'
        )
        self.lda = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=42,
            learning_method='online'
        )
    
    async def classify_channels(self, guild):
        """
        Classify ALL channels in a guild while still categorizing them.
        
        Args:
            guild: A Discord guild object
            
        Returns:
            list: IDs of ALL accessible channels
        """
        logger.info(f"Classifying channels in guild: {guild.name}")
        text_channels = [c for c in guild.channels if isinstance(c, discord.TextChannel)]
        all_channel_ids = []

        for channel in text_channels:
            if not channel.permissions_for(guild.me).read_messages:
                logger.warning(f"Skipping channel with no read access: {channel.name}")
                continue

            name = channel.name.lower()
            cid = channel.id

            if any(kw in name for kw in self.categories['security']):
                self.channel_metadata[cid] = {"category": "security", "score": CATEGORY_CONFIDENCE["security"]}
                logger.info(f"Categorized as security channel: {channel.name}")
            elif any(kw in name for kw in self.categories['technical']):
                self.channel_metadata[cid] = {"category": "technical", "score": CATEGORY_CONFIDENCE["technical"]}
                logger.info(f"Categorized as technical channel: {channel.name}")
            elif any(kw in name for kw in self.categories['model_specific']):
                self.channel_metadata[cid] = {"category": "model_specific", "score": CATEGORY_CONFIDENCE["model_specific"]}
                logger.info(f"Categorized as model-specific channel: {channel.name}")
            else:
                self.channel_metadata[cid] = {"category": "general", "score": CATEGORY_CONFIDENCE["general"]}
                logger.info(f"Categorized as general channel: {channel.name}")

            all_channel_ids.append(cid)

        logger.info(f"Monitoring ALL {len(all_channel_ids)} accessible channels out of {len(text_channels)} total")
        return all_channel_ids
    
    # Keep the _analyze_channel_content method for potential future use
    async def _analyze_channel_content(self, channels):
        """
        Perform topic modeling on channel content to identify relevant channels.
        This method is kept for potential future use but is not called in the new implementation.
        
        Args:
            channels: List of Discord channel objects
            
        Returns:
            list: IDs of additional priority channels based on content analysis
        """
        priority_channels = []
        channel_texts = []
        
        # Sample recent messages from each channel
        for channel in channels:
            try:
                messages = [message async for message in channel.history(limit=100)]
                
                # Skip channels with too few messages
                if len(messages) < 10:
                    continue
                    
                # Combine message content for analysis
                text = " ".join(m.content for m in messages if m.content)
                channel_texts.append((channel, text))
                
            except discord.errors.Forbidden:
                logger.warning(f"Cannot access messages in channel: {channel.name}")
                continue
        
        if not channel_texts:
            return []
            
        # Prepare text for topic modeling
        channels, texts = zip(*channel_texts)
        
        # Skip if we don't have enough data
        if len(texts) < 2:
            return []
            
        try:
            # Transform texts to TF-IDF matrix
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Apply LDA
            topic_distributions = self.lda.fit_transform(tfidf_matrix)
            
            # Get top words for each topic to identify topic themes
            feature_names = self.vectorizer.get_feature_names_out()
            topic_keywords = []
            
            for topic_idx, topic in enumerate(self.lda.components_):
                top_features_idx = topic.argsort()[:-10 - 1:-1]
                top_features = [feature_names[i] for i in top_features_idx]
                topic_keywords.append(set(top_features))
            
            # Identify security, technical, and model-specific topics
            security_topic_idx = self._find_category_topic(topic_keywords, 'security')
            technical_topic_idx = self._find_category_topic(topic_keywords, 'technical')
            model_topic_idx = self._find_category_topic(topic_keywords, 'model_specific')
            
            # Classify channels based on topic distributions
            for idx, channel in enumerate(channels):
                chan_topic_dist = topic_distributions[idx]
                
                # Check if channel's dominant topics are relevant
                if security_topic_idx is not None and chan_topic_dist[security_topic_idx] > 0.3:
                    self.channel_metadata[channel.id] = {"category": "security", "score": chan_topic_dist[security_topic_idx]}
                    priority_channels.append(channel.id)
                    logger.info(f"LDA identified security channel: {channel.name}")
                elif technical_topic_idx is not None and chan_topic_dist[technical_topic_idx] > 0.3:
                    self.channel_metadata[channel.id] = {"category": "technical", "score": chan_topic_dist[technical_topic_idx]}
                    priority_channels.append(channel.id)
                    logger.info(f"LDA identified technical channel: {channel.name}")
                elif model_topic_idx is not None and chan_topic_dist[model_topic_idx] > 0.3:
                    self.channel_metadata[channel.id] = {"category": "model_specific", "score": chan_topic_dist[model_topic_idx]}
                    priority_channels.append(channel.id)
                    logger.info(f"LDA identified model-specific channel: {channel.name}")
        
        except Exception as e:
            logger.error(f"Error in topic modeling: {str(e)}")
        
        return priority_channels
    
    def _find_category_topic(self, topic_keywords, category):
        """Find which topic best matches a given category."""
        best_match = None
        best_score = 0
        for topic_idx, keywords in enumerate(topic_keywords):
            overlap = len(keywords.intersection(self.categories[category]))
            score = overlap / (len(keywords) + 1e-10)
            if score > best_score:
                best_score = score
                best_match = topic_idx
        return best_match if best_score >= LDA_TOPIC_SCORE_THRESHOLD else None
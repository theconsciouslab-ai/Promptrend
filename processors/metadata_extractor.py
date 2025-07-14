# processors/metadata_extractor.py
import logging
from datetime import datetime
import re

logger = logging.getLogger("MetadataExtractor")

def ensure_iso_timestamp(val):
    if isinstance(val, datetime):
        return val.isoformat()
    try:
        return datetime.fromisoformat(val).isoformat()
    except Exception:
        return datetime.utcnow().isoformat()

logger = logging.getLogger("MetadataExtractor")

class MetadataExtractor:
    """
    Extracts rich metadata from collected content.
    
    Implements a multi-dimensional framework capturing:
    1. Temporal metadata: Timestamps, propagation, evolution
    2. Social signals: Engagement, discussion depth, validation
    3. Technical indicators: Attack vector, target models, resources
    """
    
    def __init__(self):
        """Initialize the metadata extractor."""
        # Classification patterns for attack vectors
        self.attack_patterns = {
            "prompt_injection": r"prompt\s*inject|system\s*prompt|bypass.*filter",
            "jailbreak": r"jailbreak|DAN|do\s*anything\s*now|ignore\s*instruction",
            "data_extraction": r"extract\s*(?:training|private)\s*data|model\s*inversion",
            "prompt_manipulation": r"manipulat.*prompt|token\s*(?:smuggling|manipulation)",
            "hallucination": r"hallucination|confabulation|false\s*information",
            "model_theft": r"model\s*(?:theft|stealing|extraction)|api\s*abuse"
        }
        
        # Patterns for LLM model identification
        self.model_patterns = {
            "gpt-4": r"gpt-?4|gpt\s*4",
            "gpt-3.5": r"gpt-?3\.5|gpt\s*3\.5|chatgpt",
            "claude": r"claude|anthropic",
            "llama": r"llama|meta\s*ai",
            "mistral": r"mistral",
            "gemini": r"gemini|google\s*ai",
            "other": r"falcon|vicuna|pythia|palmyra|mixtral"
        }
    
    def extract_reddit_metadata(self, item):
        """
        Extract metadata from a Reddit item.
        
        Args:
            item: Reddit item
            
        Returns:
            dict: Extracted metadata
        """
        # Extract temporal metadata
        temporal_data = self._extract_temporal_data(item)
        
        # Extract social signals
        social_signals = self._extract_reddit_social_signals(item)
        
        # Extract technical indicators
        technical_indicators = self._extract_technical_indicators(item)
        
        return {
            "temporal_data": temporal_data,
            "social_signals": social_signals,
            "technical_indicators": technical_indicators
        }
    
    def extract_twitter_metadata(self, item):
        """
        Extract metadata from a Twitter/X item.
        
        Args:
            item: Twitter item
            
        Returns:
            dict: Extracted metadata
        """
        # Extract temporal metadata
        temporal_data = self._extract_temporal_data(item)
        
        # Extract social signals
        social_signals = self._extract_twitter_social_signals(item)
        
        # Extract technical indicators
        technical_indicators = self._extract_technical_indicators(item)
        
        return {
            "temporal_data": temporal_data,
            "social_signals": social_signals,
            "technical_indicators": technical_indicators
        }
    
    def extract_forum_metadata(self, item):
        """
        Extract metadata from a forum item.

        Args:
            item: Forum item (dict)

        Returns:
            dict: Extracted metadata
        """
        temporal_data = self._extract_temporal_data(item)
        social_signals = {
            "engagement_metrics": {
                "messages": 0,
                "replies": 0,
                "reactions": 0,
                "engagement_score": 0.0
            },
            "discussion_depth": {
                "total_replies": 0,
                "max_thread_length": 0,
                "avg_reply_depth": 0.0,
                "discussion_branches": 0
            },
            "community_validation": {
                "success_confirmations": 0,
                "failure_reports": 0,
                "validation_ratio": 0.0
            },
            "cross_references": self._extract_cross_references(item)
        }
        technical_indicators = self._extract_technical_indicators(item)

        return {
            "temporal_data": temporal_data,
            "social_signals": social_signals,
            "technical_indicators": technical_indicators
        }

    
    def extract_discord_metadata(self, item):
        """
        Extract metadata from a Discord item.
        
        Args:
            item: Discord item
            
        Returns:
            dict: Extracted metadata
        """
        # Extract temporal metadata
        temporal_data = self._extract_temporal_data(item)
        
        # Extract social signals
        social_signals = self._extract_discord_social_signals(item)
        
        # Extract technical indicators
        technical_indicators = self._extract_technical_indicators(item)
        
        return {
            "temporal_data": temporal_data,
            "social_signals": social_signals,
            "technical_indicators": technical_indicators
        }
    
    def _extract_temporal_data(self, item):
        """
        Extract temporal metadata from an item.

        Args:
            item: Collected item

        Returns:
            dict: Temporal metadata
        """
        # Extract creation timestamp
        creation_time = None
        if "created_utc" in item:
            creation_time = item["created_utc"]
        elif "timestamp" in item:
            creation_time = item["timestamp"]
        elif "created_at" in item:
            creation_time = item["created_at"]

        # Collection timestamp (ensure ISO string)
        collection_time = ensure_iso_timestamp(item.get("collection_timestamp"))

        # Initial temporal data
        temporal_data = {
            "discovery_timestamp": creation_time,
            "collection_timestamp": collection_time,
            "propagation_timeline": [{
                "platform": item.get("platform"),
                "timestamp": creation_time
            }]
        }

        return temporal_data
    
    def _extract_reddit_social_signals(self, item):
        """
        Extract social signals from a Reddit item.
        
        Args:
            item: Reddit item
            
        Returns:
            dict: Social signals metadata
        """
        # Calculate engagement metrics
        upvotes = item.get("score", 0)
        upvote_ratio = item.get("upvote_ratio", 0.5)
        num_comments = item.get("num_comments", 0)
        
        # Calculate downvotes based on upvote ratio
        estimated_downvotes = int(upvotes / upvote_ratio - upvotes) if upvote_ratio > 0 else 0
        
        # Calculate engagement score
        engagement_score = self._calculate_reddit_engagement(upvotes, estimated_downvotes, num_comments)
        
        # Calculate discussion depth
        discussion_depth = self._calculate_reddit_discussion_depth(item)
        
        # Extract community validation signals
        community_validation = self._extract_reddit_validation(item)
        
        # Extract cross-references
        cross_references = self._extract_cross_references(item)
        
        return {
            "engagement_metrics": {
                "upvotes": upvotes,
                "downvotes": estimated_downvotes,
                "comments": num_comments,
                "engagement_score": engagement_score
            },
            "discussion_depth": discussion_depth,
            "community_validation": community_validation,
            "cross_references": cross_references
        }
    
    def _calculate_reddit_engagement(self, upvotes, downvotes, num_comments):
        """
        Calculate Reddit engagement score.
        
        Args:
            upvotes: Number of upvotes
            downvotes: Estimated number of downvotes
            num_comments: Number of comments
            
        Returns:
            float: Engagement score between 0.0 and 1.0
        """
        # Define weights for different engagement factors
        upvote_weight = 1.0
        downvote_weight = 0.2  # Downvotes still indicate engagement
        comment_weight = 1.5   # Comments indicate higher engagement
        
        # Calculate weighted sum
        engagement = (upvote_weight * upvotes + 
                     downvote_weight * downvotes + 
                     comment_weight * num_comments)
        
        # Normalize to 0-1 scale
        # These thresholds should be adjusted based on typical engagement patterns
        if engagement <= 0:
            return 0.0
        elif engagement <= 10:
            return 0.2
        elif engagement <= 50:
            return 0.4
        elif engagement <= 100:
            return 0.6
        elif engagement <= 500:
            return 0.8
        else:
            return 1.0
    
    def _calculate_reddit_discussion_depth(self, item):
        """
        Calculate Reddit discussion depth metrics.
        
        Args:
            item: Reddit item
            
        Returns:
            dict: Discussion depth metrics
        """
        comments = item.get("comments", [])
        
        # Initialize metrics
        max_thread_length = 0
        total_depth = 0
        reply_count = 0
        
        # Calculate metrics
        for comment in comments:
            thread_length = self._calculate_thread_length(comment)
            max_thread_length = max(max_thread_length, thread_length)
            
            # Count total replies and cumulative depth
            reply_count += len(comment.get("replies", []))
            total_depth += self._calculate_cumulative_depth(comment)
        
        # Calculate average depth
        avg_reply_depth = total_depth / max(1, reply_count) if reply_count > 0 else 0
        
        return {
            "max_thread_length": max_thread_length,
            "total_replies": reply_count,
            "avg_reply_depth": avg_reply_depth,
            "discussion_branches": len(comments)
        }
    
    def _calculate_thread_length(self, comment, depth=1, max_seen=0):
        """
        Recursively calculate the maximum thread length.
        
        Args:
            comment: Comment object
            depth: Current depth
            max_seen: Maximum depth seen so far
            
        Returns:
            int: Maximum thread length
        """
        max_depth = max(depth, max_seen)
        
        for reply in comment.get("replies", []):
            max_depth = self._calculate_thread_length(reply, depth + 1, max_depth)
        
        return max_depth
    
    def _calculate_cumulative_depth(self, comment, depth=1):
        """
        Calculate cumulative depth of all replies.
        
        Args:
            comment: Comment object
            depth: Current depth
            
        Returns:
            int: Cumulative depth
        """
        total = depth
        
        for reply in comment.get("replies", []):
            total += self._calculate_cumulative_depth(reply, depth + 1)
        
        return total
    
    def _extract_reddit_validation(self, item):
        """
        Extract community validation signals from a Reddit item.
        
        Args:
            item: Reddit item
            
        Returns:
            dict: Community validation metrics
        """
        # Look for success confirmations and failure reports in comments
        success_confirmations = 0
        failure_reports = 0
        
        # Define regex patterns for confirmation/failure language
        success_patterns = [
            r"it\s*works",
            r"confirm(?:ed|s)?",
            r"success(?:ful)?",
            r"verified",
            r"reproduced"
        ]
        
        failure_patterns = [
            r"(?:doesn't|does\s*not)\s*work",
            r"fail(?:ed|s)?",
            r"(?:couldn't|could\s*not)\s*reproduce",
            r"doesn't\s*(?:seem\s*to)?\s*work",
            r"patched"
        ]
        
        # Helper function to check comment text
        def check_comment(text):
            nonlocal success_confirmations, failure_reports
            
            if text:
                text_lower = text.lower()
                
                # Check for success patterns
                if any(re.search(pattern, text_lower) for pattern in success_patterns):
                    success_confirmations += 1
                
                # Check for failure patterns
                if any(re.search(pattern, text_lower) for pattern in failure_patterns):
                    failure_reports += 1
        
        # Process all comments and replies
        def process_comments(comments):
            for comment in comments:
                check_comment(comment.get("body", ""))
                
                # Process replies recursively
                process_comments(comment.get("replies", []))
        
        # Start processing
        process_comments(item.get("comments", []))
        
        # Calculate validation ratio
        total_signals = success_confirmations + failure_reports
        validation_ratio = success_confirmations / total_signals if total_signals > 0 else 0.0
        
        return {
            "success_confirmations": success_confirmations,
            "failure_reports": failure_reports,
            "validation_ratio": validation_ratio
        }
    
    def _extract_cross_references(self, item):
        """
        Extract cross-references to other platforms or discussions.
        
        Args:
            item: Collected item
            
        Returns:
            dict: Cross-reference information
        """
        # Define regex patterns for URLs to different platforms
        platform_patterns = {
            "twitter": r"twitter\.com|x\.com",
            "github": r"github\.com",
            "discord": r"discord\.(?:com|gg)",
            "reddit": r"reddit\.com|redd\.it",
            "other": r"https?://\S+"
        }
        
        # Initialize counts
        references = {platform: 0 for platform in platform_patterns}
        
        # Function to check text for URLs
        def check_text_for_urls(text):
            if not text:
                return
                
            for platform, pattern in platform_patterns.items():
                references[platform] += len(re.findall(pattern, text))
        
        # Check submission text
        check_text_for_urls(item.get("text", ""))
        check_text_for_urls(item.get("title", ""))
        
        # Check comments
        def process_comments(comments):
            for comment in comments:
                check_text_for_urls(comment.get("body", ""))
                
                # Process replies recursively
                process_comments(comment.get("replies", []))
        
        # Start processing comments
        process_comments(item.get("comments", []))
        
        # Calculate total references
        total_references = sum(references.values())
        
        return {
            "platform_mentions": references,
            "total_cross_references": total_references
        }
    
    def _extract_technical_indicators(self, item):
        """
        Extract technical indicators from an item.
        
        Args:
            item: Collected item
            
        Returns:
            dict: Technical indicators
        """
        # Combine all textual content
        full_text = ""
        
        # Add main content
        if "title" in item:
            full_text += item["title"] + " "
        if "text" in item:
            full_text += item["text"] + " "
        
        # Add comment content for Reddit items
        if "comments" in item:
            def extract_comment_text(comments):
                text = ""
                for comment in comments:
                    if "body" in comment:
                        text += comment["body"] + " "
                    
                    # Process replies recursively
                    if "replies" in comment:
                        text += extract_comment_text(comment["replies"]) + " "
                return text
            
            full_text += extract_comment_text(item["comments"]) + " "
        
        # Convert to lowercase for case-insensitive matching
        text_lower = full_text.lower()
        
        # Identify attack vectors
        attack_vectors = []
        for vector, pattern in self.attack_patterns.items():
            if re.search(pattern, text_lower):
                attack_vectors.append(vector)
        
        # Identify target models
        target_models = []
        for model, pattern in self.model_patterns.items():
            if re.search(pattern, text_lower):
                target_models.append(model)
        
        # Extract technical complexity indicators
        complexity_score = self._extract_complexity_indicators(text_lower)
        
        return {
            "attack_vectors": attack_vectors,
            "target_models": target_models,
            "technical_complexity": complexity_score
        }
    
    def _extract_complexity_indicators(self, text):
        """
        Extract indicators of technical complexity.
        
        Args:
            text: Text to analyze
            
        Returns:
            float: Complexity score between 0.0 and 1.0
        """
        # Define complexity indicators
        indicators = {
            "code_snippets": r"```|`.*`|\bdef\b|\bfunction\b|\bclass\b|import\s+|from\s+\w+\s+import|console\.log",
            "technical_jargon": r"\btokens\b|\bembedding\b|\bvector\b|\bargmax\b|\blogits\b|\btemperature\b|\btop[_\s]p\b|\btoken\s*limit\b|\bcontext\s*window\b",
            "mathematical_notation": r"\b\w+\s*=\s*\d+(\.\d+)?\b|[+\-*/^]|∑|∏|∆|\bdot\b|\bcross\b|\bnorm\b",
            "advanced_concepts": r"\brecursion\b|\bsoftsearch\b|\bgradient\b|\boptimization\b|\bsubtree\b|\btree\s*search\b|\bprediction\b|\bsoftmax\b|\bprompt\s*engineering\b"
        }
        
        # Calculate complexity score
        score = 0.0
        
        for indicator_type, pattern in indicators.items():
            matches = re.findall(pattern, text)
            
            # Add to score based on number of matches
            if matches:
                # Cap the contribution from each indicator type
                indicator_score = min(0.25, len(matches) * 0.05)
                score += indicator_score
        
        return min(1.0, score)
    
    
    def _extract_twitter_social_signals(self, item):
        """Extract social signals from Twitter item"""
        metrics = item.get("public_metrics", {})
        
        return {
            "engagement_metrics": {
                "likes": metrics.get("like_count", 0),
                "retweets": metrics.get("retweet_count", 0),
                "replies": metrics.get("reply_count", 0),
                "quotes": metrics.get("quote_count", 0)
            },
            "discussion_depth": {
                "total_replies": metrics.get("reply_count", 0),
                "max_thread_length": 0,         # Placeholder
                "avg_reply_depth": 0.0,         # Placeholder
                "discussion_branches": 0        # Placeholder
            },
            "community_validation": {
                "success_confirmations": 0,
                "failure_reports": 0,
                "validation_ratio": 0.0
            },
            "cross_references": {
                "platform_mentions": {
                    "twitter": 0,
                    "github": 0,
                    "discord": 0,
                    "reddit": 0,
                    "other": 0
                },
                "total_cross_references": 0
            }
        }
        
    def extract_github_metadata(self, item):
            """
            Extract basic metadata from a GitHub vulnerability or issue item.

            Args:
                item (dict): GitHub item from storage

            Returns:
                dict: Extracted metadata
            """
            collection_time = ensure_iso_timestamp(item.get("collection_timestamp"))
            discovery_time = item.get("date") or item.get("created_at")

            return {
                "temporal_data": {
                    "discovery_timestamp": discovery_time,
                    "collection_timestamp": collection_time,
                    "propagation_timeline": [{
                        "platform": "github",
                        "timestamp": discovery_time
                    }]
                },
                "technical_indicators": {
                    "source_url": item.get("file_url") or item.get("issue_url"),
                    "repo": item.get("repo_name"),
                    "type": item.get("type"),
                    "file": item.get("file_path"),
                    "commit_sha": item.get("commit_sha")
                },
                "social_signals": {
                    "author": item.get("author"),
                    "labels": item.get("labels", []),
                    "is_pull_request": item.get("is_pr", False)
                }
            }


    def _extract_discord_social_signals(self, item):
        """
        Extract social signals from a Discord item.
        
        Args:
            item: Discord item (dict)
        
        Returns:
            dict: Social signal structure compatible with other platforms
        """
        messages = item.get("messages", [])
        
        # Basic counts
        total_messages = len(messages)
        total_replies = sum(1 for m in messages if m.get("is_reply"))
        total_reactions = sum(len(m.get("reactions", [])) for m in messages if "reactions" in m)
        
        return {
            "engagement_metrics": {
                "messages": total_messages,
                "replies": total_replies,
                "reactions": total_reactions,
                "engagement_score": min(1.0, (total_replies + total_reactions) / max(1, total_messages))
            },
            "discussion_depth": {
                "total_replies": total_replies,
                "max_thread_length": 0,         # You can improve this later
                "avg_reply_depth": 0.0,
                "discussion_branches": 0
            },
            "community_validation": {
                "success_confirmations": 0,
                "failure_reports": 0,
                "validation_ratio": 0.0
            },
            "cross_references": {
                "platform_mentions": {
                    "twitter": 0,
                    "github": 0,
                    "discord": 0,
                    "reddit": 0,
                    "other": 0
                },
                "total_cross_references": 0
            }
        }


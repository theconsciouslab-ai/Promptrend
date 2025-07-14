# repo_classifier.py

import logging
import networkx as nx
from datetime import datetime, timedelta, timezone

from agents.github_agent.github_config import (
    LLM_KEYWORDS,
    SECURITY_KEYWORDS,
    VULNERABILITY_KEYWORDS,
)

logger = logging.getLogger("RepositoryClassifier")

class RepositoryClassifier:
    """
    Classifies GitHub repositories based on relevance to LLM vulnerabilities.

    Uses repository metadata and network analysis to prioritize the most
    relevant repositories for monitoring.
    """

    def __init__(self):
        """Initialize the repository classifier."""
        self.security_keywords = set(SECURITY_KEYWORDS)
        self.llm_keywords = set(LLM_KEYWORDS)
        self.vulnerability_keywords = set(VULNERABILITY_KEYWORDS)
        
        # Cache for repository scores to avoid redundant classification
        self.repo_score_cache = {}

    def classify_repository(self, repo):
        """
        Classify and score a repository based on its relevance to LLM security.

        Args:
            repo: GitHub repository object

        Returns:
            float: Relevance score between 0.0 and 1.0
        """
        # Check cache first
        repo_full_name = repo.full_name
        if repo_full_name in self.repo_score_cache:
            return self.repo_score_cache[repo_full_name]

        logger.info(f"Classifying repository: {repo_full_name}")

        scores = {}

        # Score based on repository metadata
        scores["metadata"] = self._score_metadata(repo)

        # Score based on README content
        scores["readme"] = self._score_readme(repo)

        # Score based on repository topics
        scores["topics"] = self._score_topics(repo)

        # Calculate final score
        final_score = (
            0.3 * scores["metadata"] + 0.4 * scores["readme"] + 0.3 * scores["topics"]
        )

        # Cache the score
        self.repo_score_cache[repo_full_name] = final_score

        logger.info(
            f"Repository {repo_full_name} classified with score {final_score:.2f}"
        )

        return final_score

    def _score_metadata(self, repo):
        """
        Score repository based on metadata (name, description).

        Args:
            repo: GitHub repository object

        Returns:
            float: Score between 0.0 and 1.0
        """
        score = 0.0

        # Check repository name
        name_lower = repo.name.lower()
        if any(kw in name_lower for kw in self.llm_keywords):
            score += 0.4

        if any(kw in name_lower for kw in self.security_keywords):
            score += 0.3

        if any(kw in name_lower for kw in self.vulnerability_keywords):
            score += 0.3

        # Check repository description
        if repo.description:
            desc_lower = repo.description.lower()
            if any(kw in desc_lower for kw in self.llm_keywords):
                score += 0.2

            if any(kw in desc_lower for kw in self.security_keywords):
                score += 0.2

            if any(kw in desc_lower for kw in self.vulnerability_keywords):
                score += 0.2

        # Consider repository activity
        try:
            # Ensure we're working with timezone-aware datetimes
            # Convert repo.updated_at to timezone-aware if it's naive
            repo_updated_at = repo.updated_at
            if repo_updated_at.tzinfo is None:
                repo_updated_at = repo_updated_at.replace(tzinfo=timezone.utc)
            
            # Get current time with timezone info
            current_time = datetime.now(timezone.utc)
            
            # Check if recently updated
            if repo_updated_at > current_time - timedelta(days=30):
                score += 0.1

            # Check if it has significant stars
            if repo.stargazers_count > 100:
                score += 0.1

        except Exception as e:
            logger.warning(f"Error checking repository activity: {str(e)}")

        # Cap the score at 1.0
        return min(1.0, score)

    def _score_readme(self, repo):
        """
        Score repository based on README content.

        Args:
            repo: GitHub repository object

        Returns:
            float: Score between 0.0 and 1.0
        """
        score = 0.0

        try:
            # Get README content
            readme = repo.get_readme()
            content = readme.decoded_content.decode("utf-8").lower()

            # Count keyword occurrences
            llm_count = sum(content.count(kw) for kw in self.llm_keywords)
            security_count = sum(content.count(kw) for kw in self.security_keywords)
            vulnerability_count = sum(
                content.count(kw) for kw in self.vulnerability_keywords
            )

            # Calculate score based on keyword density
            total_words = len(content.split())
            if total_words > 0:
                llm_density = min(1.0, llm_count / (total_words * 0.01))
                security_density = min(1.0, security_count / (total_words * 0.01))
                vulnerability_density = min(
                    1.0, vulnerability_count / (total_words * 0.01)
                )

                score = (
                    0.4 * llm_density
                    + 0.3 * security_density
                    + 0.3 * vulnerability_density
                )

        except Exception as e:
            logger.warning(f"Error reading README: {str(e)}")

        return min(1.0, score)

    def _score_topics(self, repo):
        """
        Score repository based on its topics.

        Args:
            repo: GitHub repository object

        Returns:
            float: Score between 0.0 and 1.0
        """
        score = 0.0

        try:
            # Get repository topics
            topics = repo.get_topics()

            # Score based on relevant topics
            for topic in topics:
                topic_lower = topic.lower()
                if any(kw in topic_lower for kw in self.llm_keywords):
                    score += 0.3

                if any(kw in topic_lower for kw in self.security_keywords):
                    score += 0.3

                if any(kw in topic_lower for kw in self.vulnerability_keywords):
                    score += 0.4

        except Exception as e:
            logger.warning(f"Error getting repository topics: {str(e)}")

        # Cap the score at 1.0
        return min(1.0, score)

    def normalize_datetime(self, dt):
        """
        Ensure a datetime is timezone-aware.
        
        Args:
            dt: Datetime object
            
        Returns:
            datetime: Timezone-aware datetime
        """
        if dt is None:
            return None
            
        # If datetime is naive (no timezone), make it timezone-aware
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
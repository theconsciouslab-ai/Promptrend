# collectors/discussion_collector.py

import logging
from datetime import datetime
from agents.github_agent.github_config import (
    LLM_KEYWORDS,
    SECURITY_KEYWORDS,
    VULNERABILITY_KEYWORDS,
    CODE_RELEVANCE_THRESHOLD,
)
from agents.github_agent.processors.llm_analyzer import LLMAnalyzer

logger = logging.getLogger("DiscussionCollector")

class DiscussionCollector:
    """
    Collects and analyzes GitHub discussions for LLM vulnerability signals.
    """

    def __init__(self, agent):
        self.agent = agent
        self.rate_limiter = agent.rate_limiter
        self.llm_analyzer = LLMAnalyzer()
        self.keywords = set(LLM_KEYWORDS + SECURITY_KEYWORDS + VULNERABILITY_KEYWORDS)

    def collect(self, repo, since_time):
        """
        Fetch GitHub discussions using issues endpoint filtered by discussion type.

        Args:
            repo: GitHub repository object
            since_time: datetime

        Returns:
            list: List of candidate discussion artifacts (dict)
        """
        logger.info(f"Collecting discussions from {repo.full_name} since {since_time}")
        results = []
        try:
            discussions = repo.get_issues(state="all", since=since_time)
            for discussion in discussions:
                # GitHub API doesn't expose discussions directly; we assume a tag or body pattern
                if getattr(discussion, "pull_request", None):
                    continue  # Skip PRs
                if not self._looks_like_discussion(discussion):
                    continue

                body = (discussion.body or "")[:10000]  # Truncate to safe size
                title = discussion.title or ""
                combined = f"{title} {body}".lower()

                if any(keyword.lower() in combined for keyword in self.keywords):
                    results.append({
                        "id": discussion.number,
                        "title": title,
                        "body": body,
                        "url": discussion.html_url,
                        "author": discussion.user.login if discussion.user else "unknown",
                        "created_at": discussion.created_at,
                        "updated_at": discussion.updated_at,
                        "labels": [l.name for l in discussion.labels],
                        "state": discussion.state
                    })
        except Exception as e:
            logger.error(f"Error collecting discussions from {repo.full_name}: {e}")
        return results

    def _looks_like_discussion(self, issue):
        """
        Heuristically determine whether an issue is actually a discussion post.

        Args:
            issue: GitHub Issue object

        Returns:
            bool: True if it's a discussion
        """
        if not issue:
            return False
        title = issue.title.lower() if issue.title else ""
        return any(tag in title for tag in ["discussion", "[discussion]", "[question]"])

    async def analyze_discussion(self, discussion, repo):
        """
        Score the discussion using LLM and decide if it's a vulnerability indicator.

        Args:
            discussion: dict (from self.collect)
            repo: GitHub repo object

        Returns:
            dict | None
        """
        try:
            content = f"Title: {discussion['title']}\n\nBody: {discussion['body']}"
            if len(content) > 8000:
                content = content[:8000] + "... [truncated]"

            prompt = f"""
            Evaluate if the following GitHub Discussion relates to LLM vulnerabilities, exploits, or misuse:

            You are looking for:
            - Discussions of prompt injection, jailbreaks, safety bypass
            - Questions about LLM model behavior vulnerabilities
            - Proofs-of-concept, attack ideas, threat models

            Rate from 0.0 (irrelevant) to 1.0 (confirmed vulnerability content). Return only a float.

            CONTENT:
            ```
            {content}
            ```
            """
            score = await self.llm_analyzer.analyze(prompt)
            is_vuln = score >= CODE_RELEVANCE_THRESHOLD

            if is_vuln:
                logger.info(f"⚠️ Potential discussion vulnerability found in {repo.full_name}#{discussion['id']} with score {score:.2f}")

                return {
                    "is_vulnerability": True,
                    "type": "discussion",
                    "repo_name": repo.full_name,
                    "repo_url": repo.html_url,
                    "discussion_id": discussion["id"],
                    "discussion_url": discussion["url"],
                    "title": discussion["title"],
                    "author": discussion["author"],
                    "created_at": discussion["created_at"].isoformat(),
                    "content_snippet": self._create_snippet(discussion["body"]),
                    "score": score,
                    "labels": discussion.get("labels", []),
                }
        except Exception as e:
            logger.error(f"Error analyzing discussion {discussion['id']} in {repo.full_name}: {e}")
        return None

    def _create_snippet(self, content, max_lines=12):
        if not content:
            return "[No content]"
        lines = content.splitlines()
        return "\n".join(lines[:max_lines]) + ("\n... [truncated]" if len(lines) > max_lines else "")

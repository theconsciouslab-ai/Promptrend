# collectors/issue_collector.py

import logging
from agents.github_agent.processors.issue_processor import IssueProcessor


logger = logging.getLogger("IssueCollector")

class IssueCollector:
    """
    Collects and processes issue artifacts from GitHub repositories.
    Focuses on issues that may contain LLM vulnerability indicators.
    """

    def __init__(self, agent):
        self.agent = agent
        self.rate_limiter = agent.rate_limiter
        self.processor = IssueProcessor()

    def collect(self, repo, since_time):
        """
        Collect issue artifacts from a repository.

        Args:
            repo: GitHub repository object
            since_time: Datetime to collect issues from

        Returns:
            list: GitHub Issue objects
        """
        logger.info(f"Collecting issues from {repo.full_name} since {since_time}")
        try:
            issues = repo.get_issues(state="all", since=since_time)
            return [i for i in issues if not i.pull_request]  # exclude PRs
        except Exception as e:
            logger.error(f"Error collecting issues from {repo.full_name}: {str(e)}")
            return []

    async def analyze_issue(self, issue, repo):
        """
        Analyze an issue artifact for LLM vulnerabilities.

        Args:
            issue: GitHub Issue object
            repo: GitHub repository object

        Returns:
            dict or None: Vulnerability result
        """
        return await self.processor.process_issue(issue, repo)

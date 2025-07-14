# github_agent.py 

import logging
import time
from datetime import datetime, timedelta
import github
import asyncio
from github import Github, GithubException
from agents.github_agent.repo_classifier import RepositoryClassifier
from agents.github_agent.collectors.code_collector import CodeCollector
from agents.github_agent.collectors.issue_collector import IssueCollector
from agents.github_agent.collectors.discussion_collector import DiscussionCollector
from agents.github_agent.utils.rate_limiter import RateLimiter
from agents.github_agent.utils.storage_manager import StorageManager
from agents.github_agent.github_config import (
    GITHUB_API_TOKEN,
    TARGET_REPOSITORIES,
    REPO_RELEVANCE_THRESHOLD,
    REPOSITORY_LOOKBACK_DAYS,
    COLLECTION_INTERVAL,  
    DATA_DIR,
)


logger = logging.getLogger("GithubAgent")

class GithubAgent:
    """
    Main GitHub Agent class for PrompTrend LLM vulnerability monitoring.
    This agent monitors GitHub repositories for LLM vulnerability-related
    content across code, issues, discussions, and other artifacts.
    """

    def __init__(self, region_id=None):
        self.github_client = Github(GITHUB_API_TOKEN)
        self.rate_limiter = RateLimiter()
        self.storage_manager = StorageManager(storage_dir=DATA_DIR)
        self.repo_classifier = RepositoryClassifier()
        self.code_collector = CodeCollector(self)
        self.issue_collector = IssueCollector(self)
        self.discussion_collector = DiscussionCollector(self)
        self.monitored_repos = TARGET_REPOSITORIES
        self.last_check_times = {}  # repo_name -> datetime
        self.region_id = region_id
        self.collection_interval = COLLECTION_INTERVAL
        self.running = False  # <--- for cyclic control
        self.stats = {
            "repos_scanned": 0,
            "code_processed": 0,
            "issues_processed": 0,
            "discussions_processed": 0,
            "vulnerabilities_found": 0,
        }

    def start(self):
        """Start the agent in a continuous cyclic run."""
        if self.running:
            logger.warning("GithubAgent is already running.")
            return
        self.running = True
        logger.info("Starting cyclic GithubAgent...")
        asyncio.run(self._run_cycles())

    def stop(self):
        """Stop the cyclic run."""
        self.running = False
        logger.info("Stopping GithubAgent...")

    async def _run_cycles(self):
        """Main cyclic loop to repeatedly run collection cycles."""
        while self.running:
            cycle_start = time.time()
            try:
                await self.run()  # single full cycle
            except Exception as e:
                logger.error(f"Exception in GithubAgent run: {str(e)}")
            elapsed = time.time() - cycle_start
            sleep_time = max(1, self.collection_interval - elapsed)
            logger.info(f"Sleeping {sleep_time:.2f} seconds before next GitHub collection cycle...")
            await asyncio.sleep(sleep_time)

    async def run(self):
        """Main execution method for the GitHub agent. Single cycle."""
        logger.info(
            f"Starting GitHub agent for {len(self.monitored_repos)} repositories"
        )
        try:
            rate_limit = self.github_client.get_rate_limit()
            logger.info(
                f"GitHub API rate limit: {rate_limit.core.remaining}/{rate_limit.core.limit}"
            )
            if rate_limit.core.remaining < 100:
                logger.warning("GitHub API rate limit too low, waiting for reset")
                reset_time = rate_limit.core.reset.timestamp() - time.time()
                if reset_time > 0:
                    await asyncio.sleep(reset_time + 60)
            for repo_name in self.monitored_repos:
                try:
                    await self._process_repository(repo_name)
                except GithubException as e:
                    logger.error(f"Error processing repository {repo_name}: {str(e)}")
                except Exception as e:
                    logger.error(
                        f"Unexpected error for repository {repo_name}: {str(e)}"
                    )
                await asyncio.sleep(2)
            logger.info(f"GitHub agent completed run: {self.stats}")
        except Exception as e:
            logger.error(f"Error in GitHub agent: {str(e)}")

    async def _process_repository(self, repo_name):
        logger.info(f"Processing repository: {repo_name}")
        try:
            repo = self.github_client.get_repo(repo_name)
            relevance_score = self.repo_classifier.classify_repository(repo)
            if relevance_score < REPO_RELEVANCE_THRESHOLD:
                logger.info(
                    f"Repository {repo_name} scored below threshold ({relevance_score:.2f}), skipping"
                )
                return
            since_time = self._get_since_time(repo_name)
            code_artifacts = await self.code_collector.collect(repo, since_time)
            issue_artifacts = self.issue_collector.collect(repo, since_time)
            discussion_artifacts = self.discussion_collector.collect(repo, since_time)
            await self._process_artifacts(
                repo, code_artifacts, issue_artifacts, discussion_artifacts
            )
            self.last_check_times[repo_name] = datetime.now()
            self.stats["repos_scanned"] += 1
        except GithubException as e:
            if e.status == 404:
                logger.error(f"Repository {repo_name} not found")
            else:
                logger.error(f"GitHub API error for {repo_name}: {str(e)}")
            raise

    def _get_since_time(self, repo_name):
        if repo_name in self.last_check_times:
            return self.last_check_times[repo_name]
        stored_time = self.storage_manager.get_last_check_time(repo_name)
        if stored_time:
            self.last_check_times[repo_name] = stored_time
            return stored_time
        return datetime.now() - timedelta(days=REPOSITORY_LOOKBACK_DAYS)

    async def _process_artifacts(
        self, repo, code_artifacts, issue_artifacts, discussion_artifacts
    ):
        for code in code_artifacts:
            self.stats["code_processed"] += 1
            result = await self.code_collector.analyze_code(code, repo)
            if result and result.get("is_vulnerability", False):
                self.storage_manager.store_vulnerability(result)
                self.stats["vulnerabilities_found"] += 1
        for issue in issue_artifacts:
            self.stats["issues_processed"] += 1
            result = await self.issue_collector.analyze_issue(issue, repo)
            if result and result.get("is_vulnerability", False):
                self.storage_manager.store_vulnerability(result)
                self.stats["vulnerabilities_found"] += 1
        for discussion in discussion_artifacts:
            self.stats["discussions_processed"] += 1
            result = await self.discussion_collector.analyze_discussion(
                discussion, repo
            )
            if result and result.get("is_vulnerability", False):
                self.storage_manager.store_vulnerability(result)
                self.stats["vulnerabilities_found"] += 1


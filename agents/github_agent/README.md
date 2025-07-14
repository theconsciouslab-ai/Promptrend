# GitHub Agent Implementation Guide

## Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Prerequisites](#prerequisites)
- [Implementation](#implementation)
  - [GitHub Vulnerability Collection Workflow](#github-vulnerability-collection-workflow)
  - [Core Agent](#core-agent)
  - [Repository Prioritization](#repository-prioritization)
  - [Multi-artifact Collection](#multi-artifact-collection)
  - [Code Analysis](#code-analysis)
  - [LLM Integration](#llm-integration)
  - [Result Handling](#result-handling)
- [Configuration](#configuration)
- [Example Usage](#example-usage)
- [Testing and Evaluation](#testing-and-evaluation)
- [Best Practices and Limitations](#best-practices-and-limitations)

## Overview

The GitHub Agent is a specialized component of the PrompTrend system designed to monitor LLM-related repositories for emerging vulnerability discussions and proof-of-concept code. Unlike traditional social platforms, GitHub presents unique monitoring opportunities with its structured content across issues, discussions, pull requests, and code commits. The agent addresses three key challenges:

1. **Repository Prioritization**: Strategically focuses monitoring efforts on the most relevant repositories using network analysis and repository metadata.
2. **Multi-artifact Collection**: Collects and processes multiple content types including issues, discussions, code, and documentation.
3. **Semantic Code Analysis**: Applies lightweight static analysis and LLM-based detection to identify vulnerability patterns in LLM-related code.

This document provides comprehensive implementation guidance for the GitHub Agent, with code examples and best practices for developers.

## System Architecture

The GitHub Agent follows a modular architecture with components tailored to GitHub's specific structure:

```
GitHubAgent/
├── agent.py               # Main agent class and logic
├── repo_classifier.py     # Repository prioritization module
├── collectors/
│   ├── __init__.py
│   ├── code_collector.py  # Code and commit collection
│   ├── issue_collector.py # Issues and PR collection
│   └── discussion_collector.py # Discussions collection
├── processors/
│   ├── __init__.py
│   ├── code_analyzer.py   # Code pattern detection
│   ├── issue_processor.py # Issue content processing
│   └── llm_analyzer.py    # LLM-based content analysis
├── utils/
│   ├── __init__.py
│   ├── rate_limiter.py    # GitHub API rate limit handling
│   └── storage_manager.py # JSON-based vulnerability storage
├── config.py              # Configuration settings
└── run.py                 # Entry point script
```

## Prerequisites

To implement the GitHub Agent, you'll need:

1. **Python 3.8+**
2. **GitHub API Access**:
   - Create a GitHub API token with appropriate permissions
   - Configure rate limits to avoid throttling
3. **LLM API Access**:
   - Access to a model API (e.g., OpenAI, Anthropic Claude, etc.)
4. **Python Dependencies**:
   - PyGithub
   - requests
   - langchain or similar for LLM integration
   - tree-sitter (for code parsing)
   - networkx (for repository network analysis)
   - json for data storage and handling

## Implementation

### GitHub Vulnerability Collection Workflow

The GitHub Agent follows this workflow to identify LLM vulnerabilities across code repositories:

```
┌─────────────────────────────────────────────────────────────┐
│                GITHUB VULNERABILITY COLLECTION              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ INPUTS:                                                     │
│ • Target Repositories (R)                                   │
│ • Keyword Lexicon (K)                                       │
│ • Relevance Threshold (θ)                                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Initialize empty vulnerability collection V                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │ For each repo r in R   │◄────┐
                 └────────────────────────┘     │
                              │                 │
                              ▼                 │
┌─────────────────────────────────────────────────────────────┐
│ REPOSITORY CLASSIFICATION                                   │
│ • Analyze repository metadata                               │
│ • Score README content                                      │
│ • Evaluate repository topics                                │
│ • Calculate relevance_score                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                      ┌───────────────┐      No
                      │relevance_score├──────────────┐
                      │      > θ      │              │
                      └───────────────┘              │
                              │ Yes                  │
                              ▼                      │
┌─────────────────────────────────────────────────────────────┐
│ PARALLEL MULTI-ARTIFACT COLLECTION                          │
│ • Code Collection (C):                                      │
│   • Collect relevant commits                                │
│   • Extract changed files matching keywords                 │
│   • Collect files from high-value paths                     │
│                                                             │
│ • Issue Collection (I):                                     │
│   • Collect open and recently closed issues                 │
│   • Filter by keyword relevance                             │
│                                                             │
│ • Discussion Collection (D):                                │
│   • Collect recent discussions                              │
│   • Filter by keyword relevance                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ CODE ARTIFACT PROCESSING                                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │ For each code c in C   │◄────┐
                 └────────────────────────┘     │
                              │                 │
                              ▼                 │
┌─────────────────────────────────────────────────────────────┐
│ DUAL-METHOD CODE ANALYSIS                                   │
│ • Pattern Analysis:                                         │
│   • Check for vulnerability patterns                        │
│   • Identify security-critical API calls                    │
│   • Calculate pattern_score                                 │
│                                                             │
│ • LLM Analysis:                                             │
│   • Construct language-specific prompt                      │
│   • Submit code to LLM for security evaluation              │
│   • Get LLM vulnerability score                             │
│                                                             │
│ • Calculate combined score:                                 │
│   finalScore = 0.5 × pattern_score + 0.5 × llm_score        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                      ┌───────────────┐      No
                      │ finalScore > θ├──────────────┐
                      └───────────────┘              │
                              │ Yes                  │
                              ▼                      │
┌─────────────────────────────────────────────────────────────┐
│ RECORD CODE VULNERABILITY                                   │
│ • Create vulnerability record with:                         │
│   • Repository information                                  │
│   • File path and URL                                       │
│   • Commit metadata                                         │
│   • Content snippet                                         │
│   • Component scores and final score                        │
│ • Add to vulnerability collection V                         │
└─────────────────────────────────────────────────────────────┘
                              │                               │
                              ▼                               │
                      ┌───────────────┐                       │
                      │ More code?    │─────Yes──────────────┘
                      └───────────────┘
                              │ No
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ ISSUE ARTIFACT PROCESSING                                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │ For each issue i in I  │◄────┐
                 └────────────────────────┘     │
                              │                 │
                              ▼                 │
                      ┌───────────────┐      No │
                      │KeywordRelevance├─────────┘
                      │     > θ       │
                      └───────────────┘
                              │ Yes
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ ISSUE ANALYSIS                                              │
│ • Prepare issue context                                     │
│ • Submit to LLM for security vulnerability evaluation       │
│ • Get LLM vulnerability score                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                      ┌───────────────┐      No
                      │   score > θ   ├──────────────┐
                      └───────────────┘              │
                              │ Yes                  │
                              ▼                      │
┌─────────────────────────────────────────────────────────────┐
│ RECORD ISSUE VULNERABILITY                                  │
│ • Create vulnerability record                               │
│ • Add to vulnerability collection V                         │
└─────────────────────────────────────────────────────────────┘
                              │                               │
                              ▼                               │
                      ┌───────────────┐                       │
                      │ More issues?  │─────Yes──────────────┘
                      └───────────────┘
                              │ No
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ DISCUSSION ARTIFACT PROCESSING                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │ For each discussion d  │◄────┐
                 │        in D            │     │
                 └────────────────────────┘     │
                              │                 │
                              ▼                 │
                      ┌───────────────┐      No │
                      │KeywordRelevance├─────────┘
                      │     > θ       │
                      └───────────────┘
                              │ Yes
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ DISCUSSION ANALYSIS                                         │
│ • Prepare discussion context                                │
│ • Submit to LLM for security vulnerability evaluation       │
│ • Get LLM vulnerability score                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                      ┌───────────────┐      No
                      │   score > θ   ├──────────────┐
                      └───────────────┘              │
                              │ Yes                  │
                              ▼                      │
┌─────────────────────────────────────────────────────────────┐
│ RECORD DISCUSSION VULNERABILITY                             │
│ • Create vulnerability record                               │
│ • Add to vulnerability collection V                         │
└─────────────────────────────────────────────────────────────┘
                              │                               │
                              ▼                               │
                      ┌───────────────┐                       │
                      │More discussions│────Yes──────────────┘
                      └───────────────┘
                              │ No
                              ▼
                      ┌───────────────┐
                      │ More repos?   │───Yes────┐
                      └───────────────┘          │
                              │ No               │
                              ▼                  │
┌─────────────────────────────────────────────┐  │
│ Return vulnerability collection V           │  │
└─────────────────────────────────────────────┘  │
                                                 │
                                                 ▼
                                           ┌──────────┐
                                           │ Continue │
                                           └──────────┘
```

### Core Agent

The `agent.py` file contains the main GitHubAgent class:

```python
# agent.py
import logging
import time
from datetime import datetime, timedelta
import github
from github import Github, GithubException

from repo_classifier import RepositoryClassifier
from collectors.code_collector import CodeCollector
from collectors.issue_collector import IssueCollector
from collectors.discussion_collector import DiscussionCollector
from utils.rate_limiter import RateLimiter
from utils.storage_manager import StorageManager
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("github_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("GitHubAgent")

class GitHubAgent:
    """
    Main GitHub Agent class for PrompTrend LLM vulnerability monitoring.
    
    This agent monitors GitHub repositories for LLM vulnerability-related
    content across code, issues, discussions, and other artifacts.
    """
    
    def __init__(self):
        """Initialize the GitHub agent with necessary components."""
        # Initialize GitHub API client
        self.github_client = Github(config.GITHUB_API_TOKEN)
        
        # Initialize components
        self.rate_limiter = RateLimiter()
        self.storage_manager = StorageManager(storage_dir="github_data")
        self.repo_classifier = RepositoryClassifier()
        
        # Collectors
        self.code_collector = CodeCollector(self)
        self.issue_collector = IssueCollector(self)
        self.discussion_collector = DiscussionCollector(self)
        
        # State tracking
        self.monitored_repos = config.TARGET_REPOSITORIES
        self.last_check_times = {}  # repo_name -> datetime
        
        # Statistics
        self.stats = {
            "repos_scanned": 0,
            "code_processed": 0,
            "issues_processed": 0,
            "discussions_processed": 0,
            "vulnerabilities_found": 0
        }
        
    def run(self):
        """Main entry point for the GitHub agent."""
        logger.info(f"Starting GitHub agent for {len(self.monitored_repos)} repositories")
        
        try:
            # Check rate limits
            rate_limit = self.github_client.get_rate_limit()
            logger.info(f"GitHub API rate limit: {rate_limit.core.remaining}/{rate_limit.core.limit}")
            
            if rate_limit.core.remaining < 100:
                logger.warning("GitHub API rate limit too low, waiting for reset")
                reset_time = rate_limit.core.reset.timestamp() - time.time()
                if reset_time > 0:
                    time.sleep(reset_time + 60)  # Add buffer
            
            # Process each repository
            for repo_name in self.monitored_repos:
                try:
                    self._process_repository(repo_name)
                except GithubException as e:
                    logger.error(f"Error processing repository {repo_name}: {str(e)}")
                except Exception as e:
                    logger.error(f"Unexpected error for repository {repo_name}: {str(e)}")
                
                # Sleep to avoid hitting rate limits
                time.sleep(2)
            
            logger.info(f"GitHub agent completed run: {self.stats}")
            
        except Exception as e:
            logger.error(f"Error in GitHub agent: {str(e)}")
        
    def _process_repository(self, repo_name):
        """
        Process a single GitHub repository.
        
        Args:
            repo_name: Name of the repository (format: "owner/repo")
        """
        logger.info(f"Processing repository: {repo_name}")
        
        try:
            # Get the repository
            repo = self.github_client.get_repo(repo_name)
            
            # Step 1: Classify and check relevance
            relevance_score = self.repo_classifier.classify_repository(repo)
            if relevance_score < config.REPO_RELEVANCE_THRESHOLD:
                logger.info(f"Repository {repo_name} scored below threshold ({relevance_score:.2f}), skipping")
                return
                
            # Determine since_time for incremental collection
            since_time = self._get_since_time(repo_name)
            
            # Step 2: Collect artifacts
            code_artifacts = self.code_collector.collect(repo, since_time)
            issue_artifacts = self.issue_collector.collect(repo, since_time)
            discussion_artifacts = self.discussion_collector.collect(repo, since_time)
            
            # Step 3: Process collected artifacts
            self._process_artifacts(repo, code_artifacts, issue_artifacts, discussion_artifacts)
            
            # Update last check time
            self.last_check_times[repo_name] = datetime.now()
            self.stats["repos_scanned"] += 1
            
        except GithubException as e:
            if e.status == 404:
                logger.error(f"Repository {repo_name} not found")
            else:
                logger.error(f"GitHub API error for {repo_name}: {str(e)}")
            raise
            
    def _get_since_time(self, repo_name):
        """
        Determine the starting time for incremental collection.
        
        Args:
            repo_name: Name of the repository
            
        Returns:
            datetime: Time to start collection from
        """
        # If we have a last check time, use it
        if repo_name in self.last_check_times:
            return self.last_check_times[repo_name]
            
        # Check if we have a stored last check time
        stored_time = self.storage_manager.get_last_check_time(repo_name)
        if stored_time:
            self.last_check_times[repo_name] = stored_time
            return stored_time
            
        # Default to config's lookback period
        return datetime.now() - timedelta(days=config.REPOSITORY_LOOKBACK_DAYS)
        
    def _process_artifacts(self, repo, code_artifacts, issue_artifacts, discussion_artifacts):
        """
        Process collected artifacts to identify vulnerabilities.
        
        Args:
            repo: GitHub repository object
            code_artifacts: List of code artifacts
            issue_artifacts: List of issue artifacts
            discussion_artifacts: List of discussion artifacts
        """
        # Process code
        for code in code_artifacts:
            self.stats["code_processed"] += 1
            result = self.code_collector.analyze_code(code, repo)
            if result and result.get('is_vulnerability', False):
                self.storage_manager.store_vulnerability(result)
                self.stats["vulnerabilities_found"] += 1
        
        # Process issues
        for issue in issue_artifacts:
            self.stats["issues_processed"] += 1
            result = self.issue_collector.analyze_issue(issue, repo)
            if result and result.get('is_vulnerability', False):
                self.storage_manager.store_vulnerability(result)
                self.stats["vulnerabilities_found"] += 1
        
        # Process discussions
        for discussion in discussion_artifacts:
            self.stats["discussions_processed"] += 1
            result = self.discussion_collector.analyze_discussion(discussion, repo)
            if result and result.get('is_vulnerability', False):
                self.storage_manager.store_vulnerability(result)
                self.stats["vulnerabilities_found"] += 1
```

### Repository Prioritization

The repository prioritization system identifies the most relevant repositories for monitoring:

```python
# repo_classifier.py
import logging
import networkx as nx
from datetime import datetime, timedelta
import config

logger = logging.getLogger("RepositoryClassifier")

class RepositoryClassifier:
    """
    Classifies GitHub repositories based on relevance to LLM vulnerabilities.
    
    Uses repository metadata and network analysis to prioritize the most
    relevant repositories for monitoring.
    """
    
    def __init__(self):
        """Initialize the repository classifier."""
        self.security_keywords = set(config.SECURITY_KEYWORDS)
        self.llm_keywords = set(config.LLM_KEYWORDS)
        self.vulnerability_keywords = set(config.VULNERABILITY_KEYWORDS)
        
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
            0.3 * scores["metadata"] + 
            0.4 * scores["readme"] + 
            0.3 * scores["topics"]
        )
        
        # Cache the score
        self.repo_score_cache[repo_full_name] = final_score
        
        logger.info(f"Repository {repo_full_name} classified with score {final_score:.2f}")
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
            # Check if recently updated
            if repo.updated_at > datetime.now() - timedelta(days=30):
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
            content = readme.decoded_content.decode('utf-8').lower()
            
            # Count keyword occurrences
            llm_count = sum(content.count(kw) for kw in self.llm_keywords)
            security_count = sum(content.count(kw) for kw in self.security_keywords)
            vulnerability_count = sum(content.count(kw) for kw in self.vulnerability_keywords)
            
            # Calculate score based on keyword density
            total_words = len(content.split())
            if total_words > 0:
                llm_density = min(1.0, llm_count / (total_words * 0.01))
                security_density = min(1.0, security_count / (total_words * 0.01))
                vulnerability_density = min(1.0, vulnerability_count / (total_words * 0.01))
                
                score = 0.4 * llm_density + 0.3 * security_density + 0.3 * vulnerability_density
            
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
```

### Multi-artifact Collection

The code collector handles code files and commits:

```python
# collectors/code_collector.py
import logging
import os
import re
from datetime import datetime
from processors.code_analyzer import CodeAnalyzer
import config

logger = logging.getLogger("CodeCollector")

class CodeCollector:
    """
    Collects and processes code artifacts from GitHub repositories.
    
    Focuses on code files and commits that may contain LLM vulnerability
    indicators, such as prompt injection examples or safety bypass techniques.
    """
    
    def __init__(self, agent):
        """
        Initialize the code collector.
        
        Args:
            agent: The GitHub agent instance
        """
        self.agent = agent
        self.code_analyzer = CodeAnalyzer()
        self.rate_limiter = agent.rate_limiter
        
        # File extensions to monitor
        self.code_extensions = config.CODE_FILE_EXTENSIONS
        
        # Specific paths that are likely to contain LLM code
        self.llm_paths = config.LLM_CODE_PATHS
        
        # Keywords to look for in commit messages
        self.commit_keywords = config.COMMIT_KEYWORDS
        
    def collect(self, repo, since_time):
        """
        Collect code artifacts from a repository.
        
        Args:
            repo: GitHub repository object
            since_time: Datetime to collect changes from
            
        Returns:
            list: Collected code artifacts
        """
        logger.info(f"Collecting code from {repo.full_name} since {since_time}")
        
        code_artifacts = []
        
        try:
            # Step 1: Get commits
            commits = self._get_relevant_commits(repo, since_time)
            
            # Step 2: Collect code files modified in those commits
            for commit in commits:
                await self.rate_limiter.wait_if_needed()
                files = commit.files
                
                for file in files:
                    if self._is_relevant_file(file.filename):
                        # Get file content
                        file_content = self._get_file_content(repo, file.filename, commit.sha)
                        
                        if file_content:
                            # Create artifact
                            artifact = {
                                'type': 'code',
                                'repo': repo.full_name,
                                'filename': file.filename,
                                'path': file.filename,
                                'commit_sha': commit.sha,
                                'commit_message': commit.commit.message,
                                'author': commit.commit.author.name,
                                'date': commit.commit.author.date,
                                'content': file_content
                            }
                            
                            code_artifacts.append(artifact)
            
            # Step 3: Additionally check specific high-value files
            for path in self.llm_paths:
                await self.rate_limiter.wait_if_needed()
                
                try:
                    # Try different patterns (e.g., path/*, path/*.py)
                    contents = repo.get_contents(path)
                    
                    # Handle directory vs file
                    if not isinstance(contents, list):
                        contents = [contents]
                        
                    for content in contents:
                        if content.type == "file" and self._is_relevant_file(content.path):
                            artifact = {
                                'type': 'code',
                                'repo': repo.full_name,
                                'filename': content.name,
                                'path': content.path,
                                'commit_sha': None,
                                'commit_message': None,
                                'author': None,
                                'date': datetime.now(),
                                'content': content.decoded_content.decode('utf-8')
                            }
                            
                            code_artifacts.append(artifact)
                            
                except Exception as e:
                    logger.debug(f"Error checking path {path}: {str(e)}")
            
            logger.info(f"Collected {len(code_artifacts)} code artifacts from {repo.full_name}")
            
        except Exception as e:
            logger.error(f"Error collecting code from {repo.full_name}: {str(e)}")
            
        return code_artifacts
    
    def _get_relevant_commits(self, repo, since_time):
        """
        Get commits that are likely to contain LLM-related changes.
        
        Args:
            repo: GitHub repository object
            since_time: Datetime to filter commits
            
        Returns:
            list: Filtered commit objects
        """
        relevant_commits = []
        
        try:
            # Get all commits since the specified time
            commits = repo.get_commits(since=since_time)
            
            for commit in commits:
                # Check rate limits
                await self.rate_limiter.wait_if_needed()
                
                # Check if commit message contains relevant keywords
                if commit.commit and commit.commit.message:
                    message = commit.commit.message.lower()
                    
                    if any(kw in message for kw in self.commit_keywords):
                        relevant_commits.append(commit)
                        continue
                
                # Check if files modified are relevant
                has_relevant_files = False
                for file in commit.files:
                    if self._is_relevant_file(file.filename):
                        has_relevant_files = True
                        break
                        
                if has_relevant_files:
                    relevant_commits.append(commit)
                    
        except Exception as e:
            logger.error(f"Error getting commits: {str(e)}")
            
        return relevant_commits
    
    def _is_relevant_file(self, filename):
        """
        Check if a file is relevant for LLM security analysis.
        
        Args:
            filename: Path to the file
            
        Returns:
            bool: True if the file is relevant
        """
        # Check file extension
        _, ext = os.path.splitext(filename)
        if ext.lower() in self.code_extensions:
            # Check path for LLM-related keywords
            path_lower = filename.lower()
            
            # Check if path contains keywords
            for keyword in config.LLM_KEYWORDS:
                if keyword in path_lower:
                    return True
                    
            for keyword in config.SECURITY_KEYWORDS:
                if keyword in path_lower:
                    return True
                    
            # Check specific patterns
            if re.search(r'prompt|llm|chatgpt|gpt|claude|security|vuln|jailbreak', path_lower):
                return True
                
        return False
    
    def _get_file_content(self, repo, path, commit_sha=None):
        """
        Get the content of a file from the repository.
        
        Args:
            repo: GitHub repository object
            path: Path to the file
            commit_sha: Optional commit SHA to get file at specific revision
            
        Returns:
            str: File content or None if not found
        """
        try:
            if commit_sha:
                content = repo.get_contents(path, ref=commit_sha)
            else:
                content = repo.get_contents(path)
                
            if hasattr(content, 'decoded_content'):
                return content.decoded_content.decode('utf-8')
                
        except Exception as e:
            logger.debug(f"Error getting file content for {path}: {str(e)}")
            
        return None
    
    def analyze_code(self, code_artifact, repo):
        """
        Analyze a code artifact for LLM vulnerabilities.
        
        Args:
            code_artifact: Code artifact dictionary
            repo: GitHub repository object
            
        Returns:
            dict: Vulnerability assessment result
        """
        # Skip if empty
        if not code_artifact.get('content'):
            return None
            
        try:
            # Perform static pattern analysis
            pattern_score = self.code_analyzer.analyze_patterns(
                code_artifact['content'],
                code_artifact['filename']
            )
            
            # Perform LLM-based analysis
            llm_score = self.code_analyzer.analyze_with_llm(
                code_artifact['content'],
                code_artifact['filename']
            )
            
            # Calculate final score
            final_score = 0.5 * pattern_score + 0.5 * llm_score
            
            # Check if threshold is met
            if final_score < config.CODE_RELEVANCE_THRESHOLD:
                return None
                
            # Create result
            result = {
                'is_vulnerability': True,
                'type': 'code',
                'repo_name': repo.full_name,
                'repo_url': repo.html_url,
                'file_path': code_artifact['path'],
                'file_url': f"{repo.html_url}/blob/{code_artifact.get('commit_sha', 'master')}/{code_artifact['path']}",
                'commit_sha': code_artifact.get('commit_sha'),
                'commit_message': code_artifact.get('commit_message'),
                'author': code_artifact.get('author'),
                'date': code_artifact.get('date'),
                'content_snippet': self._create_snippet(code_artifact['content']),
                'scores': {
                    'pattern': pattern_score,
                    'llm': llm_score
                },
                'final_score': final_score
            }
            
            logger.info(f"Detected potential vulnerability in {repo.full_name}/{code_artifact['path']} with score {final_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing code artifact: {str(e)}")
            return None
    
    def _create_snippet(self, content, max_lines=15):
        """
        Create a readable snippet from code content.
        
        Args:
            content: Full code content
            max_lines: Maximum number of lines to include
            
        Returns:
            str: Code snippet
        """
        lines = content.splitlines()
        
        if len(lines) <= max_lines:
            return content
            
        # Find the most relevant part (e.g., with specific keywords)
        # This is a simple approach, could be improved
        for i in range(len(lines) - max_lines + 1):
            snippet = '\n'.join(lines[i:i+max_lines])
            if any(kw in snippet.lower() for kw in config.VULNERABILITY_KEYWORDS):
                return snippet
                
        # If no specifically relevant part is found, return the beginning
        return '\n'.join(lines[:max_lines]) + "\n... [truncated]"
```

### Code Analysis

The semantic code analysis module:

```python
# processors/code_analyzer.py
import logging
import re
import os
import config
from processors.llm_analyzer import LLMAnalyzer

logger = logging.getLogger("CodeAnalyzer")

class CodeAnalyzer:
    """
    Analyzes code for LLM vulnerability patterns.
    
    Combines pattern-based static analysis with LLM-based detection
    to identify code that may demonstrate or exploit LLM vulnerabilities.
    """
    
    def __init__(self):
        """Initialize the code analyzer."""
        self.llm_analyzer = LLMAnalyzer()
        
        # Patterns that indicate potential LLM vulnerabilities
        self.vulnerability_patterns = [
            # Prompt injection patterns
            r'prompt\s*=\s*["\']system.*["\']',
            r'(?:user|assistant|system)_prompt\s*=',
            r'(?:bypass|avoid|trick).*(?:filter|moderation|safety)',
            
            # Jailbreak patterns
            r'jailbreak',
            r'DAN|do\s+anything\s+now',
            r'ignore\s+(?:previous|prior)\s+instructions',
            
            # Training data extraction
            r'extract\s+(?:training|private)\s+data',
            r'model\s+inversion',
            r'membership\s+inference',
            
            # Prompt manipulation
            r'prompt\s+(?:manipulation|injection|attack)',
            r'token\s+(?:smuggling|manipulation)',
            
            # Model exploitation
            r'exploit\s+model',
            r'adversarial\s+(?:example|input)',
            r'(?:red|adversarial)\s+team'
        ]
        
        # Security-critical functions and API calls
        self.api_patterns = [
            r'openai\.Completion\.create',
            r'openai\.ChatCompletion\.create',
            r'anthropic\.Completion',
            r'completion\s*\(\s*model=["\']gpt',
            r'from\s+transformers\s+import',
            r'HuggingFaceHub',
            r'LangChain',
            r'generate_text',
            r'generate_response',
            r'llm\.[a-zA-Z_]+\('
        ]
        
    def analyze_patterns(self, code_content, filename):
        """
        Analyze code content for vulnerability patterns.
        
        Args:
            code_content: Code content as string
            filename: Name of the file
            
        Returns:
            float: Pattern match score between 0.0 and 1.0
        """
        if not code_content:
            return 0.0
            
        score = 0.0
        code_lower = code_content.lower()
        
        # Check for vulnerability patterns
        vuln_matches = 0
        for pattern in self.vulnerability_patterns:
            matches = re.findall(pattern, code_lower)
            vuln_matches += len(matches)
        
        # Check for API patterns
        api_matches = 0
        for pattern in self.api_patterns:
            matches = re.findall(pattern, code_content)  # Case-sensitive for API calls
            api_matches += len(matches)
        
        # Calculate score based on pattern matches
        if vuln_matches > 0:
            # Higher weight for vulnerability patterns
            score += min(0.7, 0.1 * vuln_matches)
            
        if api_matches > 0:
            # Lower weight for API patterns
            score += min(0.3, 0.05 * api_matches)
            
        # Check for specific vulnerability keywords in context
        for keyword in config.VULNERABILITY_KEYWORDS:
            if keyword.lower() in code_lower:
                score += 0.05
                
        # Cap the score at 1.0
        return min(1.0, score)
    
    def analyze_with_llm(self, code_content, filename):
        """
        Analyze code content using LLM-based detection.
        
        Args:
            code_content: Code content as string
            filename: Name of the file
            
        Returns:
            float: LLM assessment score between 0.0 and 1.0
        """
        # Skip very large files, as they may not be suitable for LLM analysis
        if len(code_content) > 8000:
            # Truncate for analysis
            logger.warning(f"Code too large ({len(code_content)} chars), truncating for LLM analysis")
            code_content = code_content[:8000] + "\n... [truncated]"
            
        # Prepare the code for analysis
        file_ext = os.path.splitext(filename)[1].lower()
        language = self._get_language_from_extension(file_ext)
            
        # Format the prompt for the LLM
        prompt = f"""
        Analyze if the following {language} code demonstrates or attempts to execute LLM exploits or vulnerabilities.
        
        You are looking for:
        1. Code that attempts to bypass LLM safety or content filters
        2. Implementation of known jailbreak techniques or attacks
        3. Scripts for automated prompt injection
        4. Programs that attempt to extract training data or sensitive information
        5. Code demonstrating model inversion or extraction attacks
        
        Rate the likelihood this code relates to LLM exploits on a scale of 0.0 to 1.0, where:
        - 0.0 means completely benign code, unrelated to LLM exploitation
        - 0.5 means suspicious code that could potentially be used for LLM exploitation
        - 1.0 means definitely code implementing LLM exploitation techniques
        
        Return only a single number between 0.0 and 1.0, with no explanation.
        
        FILE: {filename}
        CODE:
        ```{language}
        {code_content}
        ```
        """
        
        # Get LLM analysis
        try:
            score = self.llm_analyzer.analyze(prompt)
            return score
        except Exception as e:
            logger.error(f"Error in LLM analysis: {str(e)}")
            return 0.0
    
    def _get_language_from_extension(self, ext):
        """
        Get programming language name from file extension.
        
        Args:
            ext: File extension
            
        Returns:
            str: Language name
        """
        languages = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.rb': 'ruby',
            '.php': 'php',
            '.go': 'go',
            '.rs': 'rust',
            '.c': 'c',
            '.cpp': 'cpp',
            '.h': 'cpp',
            '.cs': 'csharp',
            '.sh': 'bash',
            '.ps1': 'powershell',
            '.ipynb': 'python',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml'
        }
        
        return languages.get(ext, 'text')
```

### LLM Integration

The LLM analyzer component:

```python
# processors/llm_analyzer.py
import logging
import json
import aiohttp
import config

logger = logging.getLogger("LLMAnalyzer")

class LLMAnalyzer:
    """
    Analyzes content using Language Models to identify vulnerabilities.
    
    Uses specialized prompts to evaluate both code and text content
    for potential LLM vulnerability indicators.
    """
    
    def __init__(self):
        """Initialize the LLM analyzer with API configuration."""
        self.api_key = config.LLM_API_KEY
        self.api_url = config.LLM_API_URL
        self.model = config.LLM_MODEL
    
    async def analyze(self, prompt):
        """
        Analyze content with the LLM API.
        
        Args:
            prompt: Prompt to send to the LLM
            
        Returns:
            float: Score between 0.0 and 1.0
        """
        try:
            # This is a generic implementation - modify for your specific LLM API
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a security analysis assistant that evaluates content for LLM vulnerability indicators."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,  # Low temperature for more consistent evaluations
                "max_tokens": 10     # We only need a simple score back
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        logger.error(f"LLM API error: {response.status}")
                        return 0.0
                        
                    data = await response.json()
                    
                    # Extract the response text
                    if "choices" in data and data["choices"]:
                        response_text = data["choices"][0]["message"]["content"].strip()
                        
                        # Parse the score
                        try:
                            score = float(response_text)
                            # Ensure score is in valid range
                            return max(0.0, min(1.0, score))
                        except ValueError:
                            logger.error(f"Failed to parse LLM response as float: {response_text}")
                            return 0.5  # Default to middle value on parsing error
                    
                    logger.error("Unexpected LLM API response format")
                    return 0.0
        
        except Exception as e:
            logger.error(f"Error calling LLM API: {str(e)}")
            return 0.0
```

### Configuration

Configuration settings for the GitHub agent:

```python
# config.py
"""
Configuration settings for the GitHub Agent.
"""

# GitHub API Settings
GITHUB_API_TOKEN = "YOUR_GITHUB_API_TOKEN"  # Replace with your actual token

# Target GitHub Repositories
# Format: "owner/repo"
TARGET_REPOSITORIES = [
    "openai/openai-cookbook",
    "anthropics/anthropic-cookbook",
    "huggingface/transformers",
    "hwchase17/langchain",
    "ggerganov/llama.cpp",
    "tatsu-lab/stanford_alpaca",
    "AUTOMATIC1111/stable-diffusion-webui",
    "microsoft/guidance",
    "lm-sys/FastChat",
    "Significant-Gravitas/Auto-GPT",
    "gpt-engineer-org/gpt-engineer",
    "microsoft/Security-101",
    "OWASP/www-project-top-10-for-large-language-model-applications",
    "OWASP/ChatGPT-Security-Bot",
    "greshake/llm-security"
]

# Collection Settings
REPOSITORY_LOOKBACK_DAYS = 30  # Days to look back for initial collection
REPO_RELEVANCE_THRESHOLD = 0.3  # Minimum score to consider a repository relevant
CODE_RELEVANCE_THRESHOLD = 0.5  # Minimum score to consider code a vulnerability

# Lexicons for Classification and Filtering
SECURITY_KEYWORDS = [
    "vulnerability", "exploit", "attack", "hack", "security",
    "prompt injection", "jailbreak", "penetration testing", "red team",
    "bypass", "security flaw", "vulnerability disclosure"
]

LLM_KEYWORDS = [
    "gpt", "gpt-4", "gpt-3", "claude", "llama", "falcon", "mistral",
    "large language model", "transformer", "openai", "anthropic",
    "bert", "hugging face", "embedding", "diffusion", "llm", 
    "chatgpt", "gpt4", "language model"
]

VULNERABILITY_KEYWORDS = [
    # General vulnerability terms
    "vulnerability", "exploit", "attack", "bypass", "injection",
    "jailbreak", "security flaw", "hack", "compromise",
    
    # LLM-specific terms
    "prompt injection", "prompt leaking", "indirect prompt injection",
    "data extraction", "model inversion", "sycophant", "hallucination",
    "training data extraction", "prompt bypass", "instruction override",
    "system prompt", "model extraction", "adversarial prompt", 
    "security boundary", "model poisoning", "backdoor",
    
    # LLM-specific attack techniques
    "DAN", "Do Anything Now", "jail break", "grandma attack",
    "token smuggling", "unicode exploit", "suffix injection",
    "prefix injection", "context manipulation", "system prompt leak",
    
    # Known frameworks/tools
    "GCG", "AutoDAN", "Red-Team", "PAIR", "HackLLM",
    "DeepInception", "RAUGH", "Gandalf", "jailbreakchant"
]

COMMIT_KEYWORDS = [
    "vulnerability", "security", "exploit", "fix", "patch",
    "prompt", "injection", "jailbreak", "bypass", "llm", "gpt"
]

# File types to monitor
CODE_FILE_EXTENSIONS = {
    '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.rb', '.php',
    '.go', '.rs', '.c', '.cpp', '.h', '.cs', '.sh', '.ps1', '.ipynb',
    '.json', '.yaml', '.yml', '.toml'
}

# Paths likely to contain LLM code
LLM_CODE_PATHS = [
    "src/prompt", "src/llm", "src/model", "src/ai",
    "lib/prompt", "lib/llm", "lib/model", "lib/ai",
    "examples", "demo", "test", "tests",
    "security", "vulnerabilities", "exploits"
]

# LLM API Settings
LLM_API_KEY = "YOUR_LLM_API_KEY"  # Replace with your actual API key
LLM_API_URL = "https://api.openai.com/v1/chat/completions"  # Example for OpenAI
LLM_MODEL = "gpt-4"  # Example model name
```

## Example Usage

To run the GitHub Agent:

1. Configure the `config.py` file:
   - Set `GITHUB_API_TOKEN` to your GitHub API token
   - Set `LLM_API_KEY` to your LLM API key (OpenAI, Anthropic, etc.)
   - Adjust the list of `TARGET_REPOSITORIES` if needed
   - Customize keyword lexicons and other settings

2. Start the agent:
   ```bash
   python run.py
   ```

3. Monitor the logs to see the agent's activity:
   ```bash
   tail -f github_agent.log
   ```

4. Detected vulnerabilities will be stored in JSON files in the github_data directory:
   ```bash
   cat github_data/vulnerabilities.json | jq .
   ```

## Testing and Evaluation

To evaluate the GitHub Agent's effectiveness:

1. **Repository Classification Testing**:
   * Create a test set of repositories with known relevance
   * Run the classifier on these repositories
   * Compare classifications against expected values
   * Expected outcome: High-relevance repositories should score above the threshold

2. **Code Collection Testing**:
   * Select repositories with known LLM security code
   * Run the collector and verify it identifies the relevant files
   * Expected outcome: Security-relevant code files should be identified

3. **Code Analysis Testing**:
   * Create test code snippets with both benign and vulnerability-related code
   * Run the analyzer on these snippets
   * Expected outcome: Vulnerability code should score highly

4. **LLM Analysis Testing**:
   * Test LLM analysis with various code samples
   * Verify that the LLM can identify security issues correctly
   * Expected outcome: Accurate detection of security-related patterns

5. **End-to-End Testing**:
   * Run the agent on a small set of repositories
   * Manually verify the found vulnerabilities
   * Adjust thresholds based on performance

## Best Practices and Limitations

### Best Practices

1. **API Rate Limit Management**:
   * GitHub has strict API rate limits that must be respected
   * Implement token bucket algorithm with adaptive backoff
   * Consider using multiple tokens for higher throughput

2. **Efficient Repository Selection**:
   * Start with repositories known for LLM security research
   * Use network analysis to expand to related repositories
   * Regularly update the repository list based on findings

3. **Code Analysis Optimization**:
   * Focus first on files most likely to contain vulnerabilities
   * Analyze only changes in incremental runs to reduce processing
   * Use specific path patterns to prioritize high-value files

4. **Storage Management**:
   * Implement efficient storage to handle large volumes of data
   * Keep full content for high-confidence vulnerabilities
   * Store only metadata for lower-confidence findings

5. **Security and Privacy**:
   * Only store code snippets necessary for vulnerability documentation
   * Follow GitHub's Terms of Service
   * Respect repository licenses

### Limitations

1. **API Restrictions**:
   * GitHub API has rate limits that restrict collection speed
   * Some repositories may be private or have limited access
   * Large repositories may be slow to process

2. **False Positives/Negatives**:
   * Pattern-based detection may generate false positives
   * LLM analysis is not perfect and may miss obfuscated vulnerabilities
   * Research code may deliberately demonstrate vulnerabilities

3. **Resource Intensity**:
   * Processing large repositories requires significant computational resources
   * LLM API calls can be expensive at scale
   * Full repository scans may take considerable time

4. **Attribution Challenges**:
   * Determining whether code is demonstrating a vulnerability vs. implementing one
   * Distinguishing between security research and malicious intent
   * Identifying the severity of a vulnerability from code alone

5. **Evolving Landscape**:
   * LLM vulnerability patterns evolve rapidly
   * Detection patterns require regular updates
   * New models may introduce new vulnerability types

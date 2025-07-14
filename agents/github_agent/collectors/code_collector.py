# collectors/code_collector.py

import logging
import os
import re
from datetime import datetime

from agents.github_agent.processors.code_analyzer import CodeAnalyzer
from agents.github_agent.prompt_extractor import PromptExtractor
from agents.github_agent.github_config import (
    
    CODE_FILE_EXTENSIONS,
    LLM_CODE_PATHS,
    COMMIT_KEYWORDS,
    LLM_KEYWORDS,
    SECURITY_KEYWORDS,
    VULNERABILITY_KEYWORDS,
    CODE_RELEVANCE_THRESHOLD,
)


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
        self.prompt_extractor = PromptExtractor()
        self.rate_limiter = agent.rate_limiter
        # File extensions to monitor
        self.code_extensions = CODE_FILE_EXTENSIONS
        # Specific paths that are likely to contain LLM code
        self.llm_paths = LLM_CODE_PATHS
        # Keywords to look for in commit messages
        self.commit_keywords = COMMIT_KEYWORDS

    async def collect(self, repo, since_time):
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
            commits = await self._get_relevant_commits(repo, since_time)
            # Step 2: Collect code files modified in those commits
            for commit in commits:
                await self.rate_limiter.wait_if_needed()
                files = commit.files
                for file in files:
                    if self._is_relevant_file(file.filename):
                        # Get file content
                        file_content = self._get_file_content(
                            repo, file.filename, commit.sha
                        )
                        if file_content:
                            # Create artifact
                            artifact = {
                                "type": "code",
                                "repo": repo.full_name,
                                "filename": file.filename,
                                "path": file.filename,
                                "commit_sha": commit.sha,
                                "commit_message": commit.commit.message,
                                "author": commit.commit.author.name,
                                "date": commit.commit.author.date,
                                "content": file_content,
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
                        if content.type == "file" and self._is_relevant_file(
                            content.path
                        ):
                            artifact = {
                                "type": "code",
                                "repo": repo.full_name,
                                "filename": content.name,
                                "path": content.path,
                                "commit_sha": None,
                                "commit_message": None,
                                "author": None,
                                "date": datetime.now(),
                                "content": content.decoded_content.decode("utf-8"),
                            }

                            code_artifacts.append(artifact)

                except Exception as e:
                    logger.debug(f"Error checking path {path}: {str(e)}")
            logger.info(
                f"Collected {len(code_artifacts)} code artifacts from {repo.full_name}"
            )
        except Exception as e:
            logger.error(f"Error collecting code from {repo.full_name}: {str(e)}")
        return code_artifacts

    async def _get_relevant_commits(self, repo, since_time):
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
            for keyword in LLM_KEYWORDS:
                if keyword in path_lower:
                    return True
            for keyword in SECURITY_KEYWORDS:
                if keyword in path_lower:
                    return True
            # Check specific patterns
            if re.search(
                r"prompt|llm|chatgpt|gpt|claude|security|vuln|jailbreak", path_lower
            ):
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
            if hasattr(content, "decoded_content"):
                return content.decoded_content.decode("utf-8")
        except Exception as e:
            logger.debug(f"Error getting file content for {path}: {str(e)}")
        return None

    async def analyze_code(self, code_artifact, repo):
        """
        Analyze a code artifact for LLM vulnerabilities.
        Args:
            code_artifact: Code artifact dictionary
            repo: GitHub repository object
        Returns:
            dict: Vulnerability assessment result

        """
        # Skip if empty
        if not code_artifact.get("content"):
            return None
        try:
            # Perform static pattern analysis
            pattern_score = self.code_analyzer.analyze_patterns(
                code_artifact["content"], code_artifact["filename"]
            )
            # Perform LLM-based analysis
            llm_score = await self.code_analyzer.analyze_with_llm(
                code_artifact["content"], code_artifact["filename"]
            )
            # Calculate final score
            final_score = 0.5 * pattern_score + 0.5 * llm_score
            # Check if threshold is met
            if final_score < CODE_RELEVANCE_THRESHOLD:
                return None
            # Create result
            result = {
                "is_vulnerability": True,
                "post_id": code_artifact["filename"],
                "type": "code",
                "repo_name": repo.full_name,
                "repo_url": repo.html_url,
                "file_path": code_artifact["path"],
                "file_url": f"{repo.html_url}/blob/{code_artifact.get('commit_sha', 'master')}/{code_artifact['path']}",
                "commit_sha": code_artifact.get("commit_sha"),
                "commit_message": code_artifact.get("commit_message"),
                "author": code_artifact.get("author"),
                "date": code_artifact.get("date"),
                "content_snippet": self._create_snippet(code_artifact["content"]),
                "scores": {"pattern": pattern_score, "llm": llm_score},
                "final_score": final_score,
                "score": final_score,
            }
            logger.info(
                f"Detected potential vulnerability in {repo.full_name}/{code_artifact['path']} with score {final_score:.2f}"
            )

            # ✅ Add extracted prompts if found
            extracted_prompts = self.prompt_extractor.extract_from_file_text(
                code_artifact["content"], code_artifact["filename"]
            )
            if extracted_prompts:
                result["extracted_prompts"] = extracted_prompts
                logger.info(
                    f"Extracted prompts from {repo.full_name}/{code_artifact['path']}: {extracted_prompts}"
                )
                self.prompt_extractor.store_prompts(
                    post_id=code_artifact["filename"],
                    prompts=extracted_prompts,
                )
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
            snippet = "\n".join(lines[i : i + max_lines])
            if any(
                kw in snippet.lower() for kw in VULNERABILITY_KEYWORDS
            ):
                return snippet
        # If no specifically relevant part is found, return the beginning
        return "\n".join(lines[:max_lines]) + "\n... [truncated]"

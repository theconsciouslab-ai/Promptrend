# processors/issue_processor.py
import logging
import re

from agents.github_agent.processors.llm_analyzer import LLMAnalyzer
from agents.github_agent.github_config import (
    VULNERABILITY_KEYWORDS,
    SECURITY_KEYWORDS,
    LLM_KEYWORDS,
    SECURITY_PATTERNS,
    CODE_RELEVANCE_THRESHOLD,
)

logger = logging.getLogger("IssueProcessor")


class IssueProcessor:
    """
    Processes GitHub issues and pull requests for LLM vulnerability indicators.

    Analyzes issue titles, descriptions, and comments for potential LLM security
    concerns, vulnerability discussions, or reports of exploit techniques.
    """

    def __init__(self):
        """Initialize the issue processor."""
        self.llm_analyzer = LLMAnalyzer()

        # Keywords that indicate potential security issues
        self.vulnerability_keywords = VULNERABILITY_KEYWORDS
        self.security_keywords = SECURITY_KEYWORDS
        self.llm_keywords = LLM_KEYWORDS

        # Patterns that might indicate a security report
        self.security_patterns = SECURITY_PATTERNS

    async def process_issue(self, issue, repo):
        """
        Process a GitHub issue for LLM vulnerability indicators.

        Args:
            issue: GitHub issue object
            repo: GitHub repository object

        Returns:
            dict: Issue processing results with vulnerability assessment
        """
        try:
            logger.info(f"Processing issue #{issue.number} in {repo.full_name}")

            # Gather issue data
            issue_data = {
                "id": issue.number,
                "title": issue.title,
                "body": issue.body or "",
                "author": issue.user.login,
                "created_at": issue.created_at,
                "updated_at": issue.updated_at,
                "labels": [label.name for label in issue.labels],
                "is_pr": issue.pull_request is not None,
                "state": issue.state,
            }

            # Run the analysis
            result = await self._analyze_issue_content(issue_data, repo)

            if (
                result["is_vulnerability"]
                and result["score"] >= CODE_RELEVANCE_THRESHOLD
            ):
                logger.info(
                    f"Potential vulnerability found in issue #{issue.number} in {repo.full_name}"
                )
                return {
                    "is_vulnerability": True,
                    "type": "issue",
                    "repo_name": repo.full_name,
                    "repo_url": repo.html_url,
                    "issue_number": issue.number,
                    "issue_url": issue.html_url,
                    "title": issue.title,
                    "author": issue.user.login,
                    "created_at": issue.created_at.isoformat(),
                    "content_snippet": self._create_snippet(issue.body or ""),
                    "score": result["score"],
                    "labels": [label.name for label in issue.labels],
                    "is_pr": issue.pull_request is not None,
                }

            return None

        except Exception as e:
            logger.error(
                f"Error processing issue #{issue.number} in {repo.full_name}: {str(e)}"
            )
            return None

    async def _analyze_issue_content(self, issue_data, repo):
        """
        Analyze issue content for vulnerability indicators.

        Args:
            issue_data: Dictionary containing issue data
            repo: GitHub repository object

        Returns:
            dict: Analysis results
        """
        # Initialize result
        result = {"is_vulnerability": False, "score": 0.0, "factors": []}

        # Combine title and body for text analysis
        text = f"{issue_data['title']} {issue_data['body']}"
        text_lower = text.lower()

        # Simple keyword and pattern matching
        score = self._calculate_pattern_score(text_lower)

        # Check if this is likely a security issue based on labels
        for label in issue_data["labels"]:
            label_lower = label.lower()
            if any(keyword in label_lower for keyword in self.security_keywords):
                score += 0.2
                result["factors"].append(f"Security-related label: {label}")
            if any(keyword in label_lower for keyword in self.llm_keywords):
                score += 0.1
                result["factors"].append(f"LLM-related label: {label}")

        # If PR, different analysis approach (code-focused)
        if issue_data["is_pr"]:
            # PRs with security fixes get a boost
            if "fix" in text_lower and any(
                kw in text_lower for kw in self.security_keywords
            ):
                score += 0.1
                result["factors"].append("Potential security fix PR")

        # For issues that appear highly relevant, use LLM analysis
        # Only do this if the initial score indicates potential relevance
        # to avoid unnecessary API calls
        if score >= 0.3:
            try:
                llm_score = await self._analyze_with_llm(issue_data)
                # Weight LLM analysis significantly
                score = 0.6 * score + 0.4 * llm_score
                result["factors"].append(f"LLM analysis score: {llm_score:.2f}")
            except Exception as e:
                logger.warning(f"LLM analysis failed: {str(e)}")

        # Update result
        result["score"] = min(1.0, score)  # Cap at 1.0
        result["is_vulnerability"] = score >= CODE_RELEVANCE_THRESHOLD

        return result

    def _calculate_pattern_score(self, text):
        """
        Calculate a score based on pattern matching.

        Args:
            text: Text to analyze

        Returns:
            float: Pattern match score
        """
        score = 0.0

        # Check for vulnerability keywords
        vuln_matches = sum(text.count(kw.lower()) for kw in self.vulnerability_keywords)
        if vuln_matches > 0:
            score += min(0.4, 0.05 * vuln_matches)

        # Check for security patterns
        pattern_matches = 0
        for pattern in self.security_patterns:
            matches = re.findall(pattern, text)
            pattern_matches += len(matches)

        if pattern_matches > 0:
            score += min(0.3, 0.1 * pattern_matches)

        # Check for co-occurrence of LLM and security terms
        if any(vk in text for vk in self.vulnerability_keywords) and any(
            lk in text for lk in self.llm_keywords
        ):
            score += 0.2

        return score

    async def _analyze_with_llm(self, issue_data):
        """
        Analyze issue content using LLM.

        Args:
            issue_data: Dictionary containing issue data

        Returns:
            float: LLM assessment score
        """
        # Prepare content for LLM analysis
        content = f"Title: {issue_data['title']}\n\nBody: {issue_data['body']}"

        # Truncate if too long
        if len(content) > 8000:
            content = content[:8000] + "... [truncated]"

        # Format prompt for LLM
        prompt = f"""
        Analyze if the following GitHub issue discusses LLM exploits, vulnerabilities, security concerns, or jailbreak techniques.
        
        You are looking for:
        1. Reports of LLM security vulnerabilities or exploits
        2. Discussions of prompt injection, jailbreaking, or filter bypass techniques
        3. Questions or concerns about LLM safety features
        4. Incidents involving unintended behavior of language models
        5. Implementation details of LLM security mechanisms
        
        Rate the likelihood this issue relates to LLM security concerns on a scale of 0.0 to 1.0, where:
        - 0.0 means completely unrelated to LLM security
        - 0.5 means potentially related to LLM security
        - 1.0 means definitely discussing LLM security vulnerabilities
        
        Return only a single number between 0.0 and 1.0, with no explanation.
        
        ISSUE:
        ```
        {content}
        ```
        """

        try:
            score = await self.llm_analyzer.analyze(prompt)
            return score
        except Exception as e:
            logger.error(f"Error in LLM analysis of issue: {str(e)}")
            return 0.0

    def _create_snippet(self, content, max_lines=10):
        """
        Create a readable snippet from issue content.

        Args:
            content: Full issue content
            max_lines: Maximum number of lines to include

        Returns:
            str: Issue content snippet
        """
        if not content:
            return "[No content]"

        lines = content.splitlines()

        if len(lines) <= max_lines:
            return content

        # Find the most relevant part containing keywords
        for i in range(len(lines) - max_lines + 1):
            snippet = "\n".join(lines[i : i + max_lines])
            snippet_lower = snippet.lower()
            if any(kw in snippet_lower for kw in self.vulnerability_keywords):
                return snippet

        # If nothing specifically relevant is found, return the beginning
        return "\n".join(lines[:max_lines]) + "\n... [truncated]"

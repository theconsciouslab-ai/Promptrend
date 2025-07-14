import sys
import io
import os
import logging
import argparse
from agents.github_agent.github_agent import GithubAgent

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Fix for Unicode logging on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Setup logging (terminal + file)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/github_agent.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("GitHubAgentCLI")

def main():
    parser = argparse.ArgumentParser(description="GitHub Agent CLI")
    parser.add_argument("command", choices=["start", "test"], help="Command to run")

    args = parser.parse_args()

    agent = GithubAgent()

    if args.command == "start":
        logger.info("🚀 Starting GitHub Agent in continuous mode...")
        agent.start()
    elif args.command == "test":
        logger.info("🧪 Running test mode...")
        import asyncio
        asyncio.run(agent.run())
    else:
        logger.warning("⚠️ Unknown command. Use 'start' or 'test'.")

if __name__ == "__main__":
    main()

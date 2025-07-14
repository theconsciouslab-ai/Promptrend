# agents/discord_agent/cli.py

import sys
import io
import logging
import asyncio
import argparse
from agents.discord_agent.discord_agent import DiscordAgent
from agents.discord_agent.discord_config import BOT_TOKEN, LOG_FILE, LOG_LEVEL

# Fix for Unicode logging on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logger = logging.getLogger("DiscordAgentCLI")

def main():
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    parser = argparse.ArgumentParser(description="Discord Agent CLI")
    parser.add_argument("command", choices=["start", "test"], help="Command to run")

    args = parser.parse_args()
    agent = DiscordAgent()

    if args.command == "start":
        logger.info("🚀 Starting Discord Agent...")
        asyncio.run(agent.start(BOT_TOKEN))

    elif args.command == "test":
        logger.info("🧪 Test mode is not yet implemented.")
        print("Implement agent.test_run() if you want test mode.")

    else:
        logger.warning("Unknown command. Use 'start' or 'test'.")

if __name__ == "__main__":
    main()

# cli.py

import asyncio
import logging
import sys
import argparse

from agents.forum_agent.forum_config import DISCUSSION_FORUMS_CYCLE_SECONDS
from agents.forum_agent.forum_agent import ForumAgent

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - PrompTrend - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/forum_agent.log", encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

async def run_once():
    setup_logging()
    logging.info("🚀 Running Discussion Forums Agent once...")
    agent = ForumAgent()
    await agent.run()

async def run_cycle():
    setup_logging()
    logger = logging.getLogger("PrompTrend")
    logger.info("🔁 Running Discussion Forums Agent in cyclic mode...")

    interval = DISCUSSION_FORUMS_CYCLE_SECONDS
    agent = ForumAgent()

    while True:
        try:
            logger.info("🚀 Starting new cycle...")
            await agent.run()
            logger.info(f"✅ Cycle completed. Sleeping for {interval} seconds...")
        except Exception as e:
            logger.exception(f"❌ Error during cycle: {e}")
        await asyncio.sleep(interval)

def main():
    parser = argparse.ArgumentParser(description="Discussion Forums Agent CLI")
    parser.add_argument("mode", choices=["start", "once"], help="Run mode: start (loop) or once (single run)")
    args = parser.parse_args()

    if args.mode == "start":
        asyncio.run(run_cycle())
    elif args.mode == "once":
        asyncio.run(run_once())

if __name__ == "__main__":
    main()

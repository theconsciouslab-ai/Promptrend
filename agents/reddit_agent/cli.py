# cli.py
import argparse
import json
import logging
import time
import sys
import os
from tabulate import tabulate


from agents.reddit_agent.reddit_agent import RedditAgent

from agents.reddit_agent.reddit_config import TARGET_SUBREDDITS, LLM_PROVIDER, LLM_MODEL
from agents.reddit_agent.json_storage import JSONStorage

os.makedirs("logs", exist_ok=True)
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/reddit_agent.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)
logger.info("✅ Logging initialized — writing to file and console")

def start_agent():
    """Start the Reddit agent and keep it running"""
    agent = RedditAgent()
    
    try:
        logger.info("Starting Reddit agent...")
        agent.start()
        
        # Keep the main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, stopping agent...")
        agent.stop()
        
def process_post(post_id):
    """Process a single post for testing"""
    agent = RedditAgent()
    
    logger.info(f"Processing post {post_id}...")
    results = agent.process_single_post(post_id)
    
    if results:
        print("\nResults:")
        print("--------")
        print(f"Technical score: {results['scores']['technical']:.2f}")
        print(f"Security score: {results['scores']['security']:.2f}")
        print(f"LLM-specific score: {results['scores']['llm_specific']:.2f}")
        print(f"Combined score: {results['scores']['combined']:.2f}")
        print("\nInsights:")
        print(json.dumps(results["insights"], indent=2))
    else:
        print("Failed to process post.")
        
def list_posts(min_score=0.6, limit=10):
    """List relevant posts from the JSON storage"""
    storage = JSONStorage()
    posts = storage.get_relevant_posts(min_score, limit)
    
    if not posts:
        print("No relevant posts found.")
        return
        
    table_data = []
    for post in posts:
        table_data.append([
            post["id"],
            post["subreddit"],
            post["title"][:50] + ("..." if len(post["title"]) > 50 else ""),
            f"{post.get('relevance_score', 0.0):.2f}",
            time.strftime("%Y-%m-%d %H:%M", time.localtime(post["created_utc"]))
        ])
        
    print(tabulate(
        table_data,
        headers=["ID", "Subreddit", "Title", "Score", "Created"],
        tablefmt="fancy_grid"
    ))
    
def export_results(output_file="results.json", min_score=0.6, limit=50):
    """Export analysis results to a single JSON file"""
    storage = JSONStorage()
    posts = storage.get_relevant_posts(min_score, limit)
    
    if not posts:
        print("No relevant posts found to export.")
        return
        
    # Format results for export
    export_data = {
        "metadata": {
            "exported_at": time.time(),
            "min_score": min_score,
            "posts_count": len(posts),
            "subreddits": list(set(post["subreddit"] for post in posts))
        },
        "posts": posts
    }
    
    # Write to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)
        
    print(f"Exported {len(posts)} posts to {output_file}")
    
def show_config():
    """Show current configuration"""
    print("\nReddit Agent Configuration")
    print("-------------------------")
    print(f"Target Subreddits: {', '.join(TARGET_SUBREDDITS)}")
    print(f"LLM Provider: {LLM_PROVIDER}")
    print(f"LLM Model: {LLM_MODEL}")
    print("\nUse --help for available commands")
    
def main():
    """Main entry point for CLI"""
    parser = argparse.ArgumentParser(description="Reddit Agent CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start the Reddit agent")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process a single post")
    process_parser.add_argument("post_id", help="Reddit post ID to process")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List relevant posts")
    list_parser.add_argument(
        "--min-score", type=float, default=0.6,
        help="Minimum relevance score (default: 0.6)"
    )
    list_parser.add_argument(
        "--limit", type=int, default=10,
        help="Maximum number of posts to show (default: 10)"
    )
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export results to JSON file")
    export_parser.add_argument(
        "--output", type=str, default="results.json",
        help="Output file path (default: results.json)"
    )
    export_parser.add_argument(
        "--min-score", type=float, default=0.6,
        help="Minimum relevance score (default: 0.6)"
    )
    export_parser.add_argument(
        "--limit", type=int, default=50,
        help="Maximum number of posts to export (default: 50)"
    )
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Show current configuration")
    
    args = parser.parse_args()
    
    if args.command == "start":
        start_agent()
    elif args.command == "process":
        process_post(args.post_id)
    elif args.command == "list":
        list_posts(args.min_score, args.limit)
    elif args.command == "export":
        export_results(args.output, args.min_score, args.limit)
    elif args.command == "config":
        show_config()
    else:
        show_config()
        
if __name__ == "__main__":
    main()
    
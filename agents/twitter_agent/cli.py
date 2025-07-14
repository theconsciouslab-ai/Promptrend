# cli.py
import argparse
import json
import logging
import time
from tabulate import tabulate

from agents.twitter_agent.twitter_agent import TwitterAgent
from agents.twitter_agent.twitter_config import TWITTER_BEARER_TOKEN, LLM_PROVIDER, LLM_MODEL
from agents.twitter_agent.json_storage import JSONStorage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def start_agent():
    """Start the Twitter agent and keep it running"""
    agent = TwitterAgent()
    
    try:
        logger.info("Starting Twitter agent...")
        agent.start()
        
        # Keep the main thread alive
        while True:
            time.sleep(60)
            
            # Log statistics every hour
            if time.time() % 3600 < 60:  # Approximately every hour
                stats = agent.stats
                logger.info(f"Stats - Collected: {stats['tweets_collected']}, "
                           f"Analyzed: {stats['tweets_analyzed']}, "
                           f"Relevant: {stats['relevant_found']}, "
                           f"Keywords added: {stats['keywords_added']}")
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, stopping agent...")
        agent.stop()
        
def process_tweet(tweet_id):
    """Process a single tweet for testing"""
    agent = TwitterAgent()
    
    logger.info(f"Processing tweet {tweet_id}...")
    results = agent.process_single_tweet(tweet_id)
    
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
        print("Failed to process tweet.")
        
def list_tweets(min_score=0.6, limit=10):
    """List relevant tweets from the JSON storage"""
    storage = JSONStorage()
    tweets = storage.get_relevant_tweets(min_score, limit)
    
    if not tweets:
        print("No relevant tweets found.")
        return
        
    table_data = []
    for entry in tweets:
        tweet = entry.get("tweet", {})
        analysis = entry.get("analysis", {})
        
        text = tweet.get("text", "")
        if len(text) > 50:
            text = text[:47] + "..."
            
        table_data.append([
            tweet.get("id", ""),
            entry.get("relevance_score", 0.0),
            text,
            analysis.get("insights", {}).get("vulnerability_type", ""),
            time.strftime("%Y-%m-%d %H:%M", time.localtime(tweet.get("created_at", 0)))
        ])
        
    print(tabulate(
        table_data,
        headers=["Tweet ID", "Score", "Content", "Vulnerability Type", "Created"],
        tablefmt="fancy_grid"
    ))
    
def export_results(output_file="results.json", min_score=0.6, limit=50):
    """Export analysis results to a single JSON file"""
    storage = JSONStorage()
    tweets = storage.get_relevant_tweets(min_score, limit)
    
    if not tweets:
        print("No relevant tweets found to export.")
        return
        
    # Format results for export
    export_data = {
        "metadata": {
            "exported_at": time.time(),
            "min_score": min_score,
            "tweets_count": len(tweets),
            "llm_provider": LLM_PROVIDER,
            "llm_model": LLM_MODEL
        },
        "tweets": tweets
    }
    
    # Write to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)
        
    print(f"Exported {len(tweets)} tweets to {output_file}")
    
def show_lexicon():
    """Show current keyword lexicon"""
    storage = JSONStorage()
    lexicon = storage.get_lexicon()
    
    print("\nCurrent Keyword Lexicon")
    print("-----------------------")
    
    # Group terms by length for better readability
    short_terms = [term for term in lexicon if len(term) < 15]
    long_terms = [term for term in lexicon if len(term) >= 15]
    
    # Print short terms in columns
    rows = []
    for i in range(0, len(short_terms), 3):
        row = short_terms[i:i+3]
        while len(row) < 3:
            row.append("")
        rows.append(row)
            
    print(tabulate(rows, tablefmt="simple"))
    
    # Print long terms on separate lines
    if long_terms:
        print("\nComplex terms:")
        for term in long_terms:
            print(f"- {term}")
    
    print(f"\nTotal: {len(lexicon)} terms")
    
def show_config():
    """Show current configuration"""
    print("\nTwitter Agent Configuration")
    print("---------------------------")
    print(f"Twitter API: {'Configured' if TWITTER_BEARER_TOKEN else 'Not Configured'}")
    print(f"LLM Provider: {LLM_PROVIDER}")
    print(f"LLM Model: {LLM_MODEL}")
    print("\nUse --help for available commands")
    
def main():
    """Main entry point for CLI"""
    parser = argparse.ArgumentParser(description="Twitter Agent CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start the Twitter agent")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process a single tweet")
    process_parser.add_argument("tweet_id", help="Twitter tweet ID to process")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List relevant tweets")
    list_parser.add_argument(
        "--min-score", type=float, default=0.6,
        help="Minimum relevance score (default: 0.6)"
    )
    list_parser.add_argument(
        "--limit", type=int, default=10,
        help="Maximum number of tweets to show (default: 10)"
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
        help="Maximum number of tweets to export (default: 50)"
    )
    
    # Lexicon command
    lexicon_parser = subparsers.add_parser("lexicon", help="Show current keyword lexicon")
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Show current configuration")
    
    args = parser.parse_args()
    
    if args.command == "start":
        start_agent()
    elif args.command == "process":
        process_tweet(args.tweet_id)
    elif args.command == "list":
        list_tweets(args.min_score, args.limit)
    elif args.command == "export":
        export_results(args.output, args.min_score, args.limit)
    elif args.command == "lexicon":
        show_lexicon()
    elif args.command == "config":
        show_config()
    else:
        show_config()
        
if __name__ == "__main__":
    main()
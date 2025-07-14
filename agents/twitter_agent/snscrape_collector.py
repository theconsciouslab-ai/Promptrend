# snscrape_collector.py
import snscrape.modules.twitter as sntwitter
import time
from agents.twitter_agent.json_storage import JSONStorage
from agents.twitter_agent.llm_analyzer import LLMAnalyzer
from agents.twitter_agent.prompt_extractor import PromptExtractor

storage = JSONStorage()
llm = LLMAnalyzer()
extractor = PromptExtractor()

query = "jailbreak prompt OR ignore instructions OR DAN -filter:retweets"
max_tweets = 10

for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
    if i >= max_tweets:
        break
    
    tweet_data = {
        "id": str(tweet.id),
        "text": tweet.content,
        "author_id": tweet.user.username,
        "created_at": tweet.date.isoformat()
    }

    print(f"Processing tweet {tweet.id}...")

    # Store tweet
    storage.store_tweet(tweet_data)

    # Fake conversation: single tweet context
    conversation = {tweet.id: tweet_data}

    # Analyze
    analysis = llm.analyze_content(tweet_data, conversation)
    storage.store_analysis(tweet_data["id"], analysis)

    # Extract prompts
    prompts = extractor.extract_prompts(tweet_data, conversation, analysis)
    extractor.store_prompts(tweet_data["id"], prompts)

import logging
import networkx as nx
import tweepy
import time
from collections import defaultdict
from agents.twitter_agent.twitter_client import TwitterClient 
from agents.twitter_agent.twitter_config import TWITTER_BEARER_TOKEN, SEED_USERS, MAX_USERS

# Configure logging
logger = logging.getLogger(__name__)

class GraphAnalyzer:
    """
    Analyzes Twitter user relationships using graph centrality measures
    to identify influential security researchers and AI practitioners.
    """
    
    def __init__(self, bearer_token=None):
        """
        Initialize the graph analyzer with Twitter API credentials
        
        Args:
            bearer_token (str, optional): Twitter API bearer token
        """
        self.bearer_token = bearer_token or TWITTER_BEARER_TOKEN
        
        
        # Initialize Tweepy client
        self.client = tweepy.Client(bearer_token=self.bearer_token)
        self.twitter_client = TwitterClient(bearer_token=self.bearer_token)
        
        # Initialize graph
        self.graph = nx.DiGraph()
        
        logger.info("Graph analyzer initialized")
        
        
    def build_graph(self, seed_users=None, max_users=MAX_USERS, depth=2):
        """
        Build a directed graph of Twitter users based on following relationships

        Args:
            seed_users (list, optional): Initial set of users to start graph building
            max_users (int, optional): Maximum number of users to include in graph
            depth (int, optional): Depth of traversal from seed users

        Returns:
            nx.DiGraph: The constructed graph
        """
        logger.info(f"Building graph from {len(seed_users or SEED_USERS)} seed users")

        # Cold-start throttle
        logger.info("Sleeping 10s before starting crawl to reduce cold-start burst")
        time.sleep(10)

        seed_users = seed_users or SEED_USERS
        visited = set()
        to_visit = [(user, 0) for user in seed_users]  # (username, depth)

        # Add seed users to graph
        for user in seed_users:
            self.graph.add_node(user, seed=True)

        # Breadth-first traversal
        while to_visit and len(self.graph) < max_users:
            current_user, current_depth = to_visit.pop(0)

            if current_user in visited:
                continue

            visited.add(current_user)

            # ✅ Step 1: Get user details with backoff
            try:
                user_info = self.client.get_user(username=current_user)
                if not user_info.data:
                    continue
                time.sleep(1)  # throttle user lookup
            except tweepy.TooManyRequests:
                self.twitter_client.handle_rate_limit()
                to_visit.append((current_user, current_depth))  # Retry later
                continue
            except Exception as e:
                logger.error(f"Error fetching user {current_user}: {str(e)}")
                continue


            user_id = user_info.data.id
            following = []

            # ✅ Step 2: Get following list with rate control
            try:
                for response in tweepy.Paginator(
                    self.client.get_users_following,
                    id=user_id,
                    max_results=100,
                    limit=5
                ):
                    if response.data:
                        following.extend([user.username for user in response.data])
                    time.sleep(1)  # throttle between page calls
            except tweepy.TooManyRequests:
                self.twitter_client.handle_rate_limit()
                to_visit.append((current_user, current_depth))  # Retry later
                continue
            except Exception as e:
                logger.error(f"Error fetching followings of {current_user}: {str(e)}")
                continue


            # Step 3: Build graph
            for followed_user in following:
                if followed_user not in self.graph:
                    self.graph.add_node(followed_user)
                self.graph.add_edge(current_user, followed_user)

            # Step 4: Add next-level users to visit queue
            if current_depth < depth:
                for followed_user in following:
                    if followed_user not in visited:
                        to_visit.append((followed_user, current_depth + 1))

            logger.info(f"Processed user {current_user}: {len(following)} following")
            time.sleep(1.5)  # throttle between users

        logger.info(f"Graph building completed: {len(self.graph)} users, {self.graph.number_of_edges()} connections")
        return self.graph

        

    def calculate_centrality(self):
        """
        Calculate various centrality measures for the graph
        
        Returns:
            dict: Dictionary mapping usernames to combined centrality scores
        """
        if not self.graph or self.graph.number_of_nodes() == 0:
            logger.warning("Graph is empty, cannot calculate centrality")
            return {}
            
        logger.info("Calculating centrality measures")
        
        # Calculate different centrality measures
        degree_centrality = nx.degree_centrality(self.graph)
        in_degree_centrality = nx.in_degree_centrality(self.graph)

        try:
            eigenvector_centrality = nx.eigenvector_centrality_numpy(self.graph, weight=None)
        except Exception:
            logger.warning("Eigenvector centrality failed — using 0s.")
            eigenvector_centrality = {node: 0.0 for node in self.graph.nodes()}

        try:
            pagerank = nx.pagerank(self.graph, alpha=0.85)
        except Exception:
            logger.warning("PageRank calculation failed — using 0s.")
            pagerank = {node: 0.0 for node in self.graph.nodes()}
        
        # Combine centrality scores with weights
        combined_scores = {}
        
        for user in self.graph.nodes():
            combined_scores[user] = (
                0.2 * degree_centrality.get(user, 0) +
                0.3 * in_degree_centrality.get(user, 0) +
                0.2 * eigenvector_centrality.get(user, 0) +
                0.3 * pagerank.get(user, 0)
            )
            
            if self.graph.nodes[user].get('seed', False):
                combined_scores[user] *= 1.2
                
        logger.info("Centrality calculation completed")
        return combined_scores

    
    
    def get_top_users(self, limit=MAX_USERS):
        """
        Get top users ranked by centrality
        
        Args:
            limit (int): Maximum number of users to return
            
        Returns:
            list: List of (username, score) tuples sorted by score
        """
        combined_scores = self.calculate_centrality()
        
        # Sort users by combined score
        ranked_users = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        logger.info(f"Generated ranked list of {len(ranked_users)} users")
        return ranked_users[:limit]

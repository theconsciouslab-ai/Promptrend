# agents/regional_coordinator.py
import logging
import importlib


logger = logging.getLogger("RegionalCoordinator")

class RegionalCoordinator:
    """
    Regional coordinator for managing platform-specific agents.
    
    Provides load balancing and coordination for a group of platform agents.
    """
    
    def __init__(self, region_id, platforms):
        """
        Initialize the regional coordinator.
        
        Args:
            region_id: Unique identifier for this region
            platforms: List of platforms to monitor in this region
        """
        self.region_id = region_id
        self.platforms = platforms
        self.platform_agents = {}
    
    def initialize_platform_agents(self):
        """Initialize platform-specific agents for this region."""
        for platform in self.platforms:
            try:
                # Dynamically import the agent class for this platform
                module_name = f"agents.{platform}_agent"
                agent_module = importlib.import_module(module_name)
                
                # Get the appropriate agent class (assuming naming convention)
                class_name = platform.capitalize() + "Agent"
                agent_class = getattr(agent_module, class_name)
                
                # Initialize the agent
                agent = agent_class(region_id=self.region_id)
                self.platform_agents[platform] = agent
                
                logger.info(f"Initialized {platform} agent for region {self.region_id}")
                
            except Exception as e:
                logger.error(f"Error initializing {platform} agent: {str(e)}")
    
    def collect_raw_data(self):
        """
        Collect raw data from all platform agents.
        
        Returns:
            list: Raw collected items
        """
        all_items = []
        
        for platform, agent in self.platform_agents.items():
            try:
                logger.info(f"Collecting from {platform} agent in region {self.region_id}")
                items = agent.collect()
                
                # Tag items with source region for routing
                for item in items:
                    item["collected_by_region"] = self.region_id
                    item["platform"] = platform
                
                all_items.extend(items)
                logger.info(f"Collected {len(items)} items from {platform}")
                
            except Exception as e:
                logger.error(f"Error collecting from {platform} agent: {str(e)}")
        
        return all_items
    
    def enrich_item(self, item):
        """
        Enrich a single item with metadata.
        
        Args:
            item: Raw collected item
            
        Returns:
            dict: Enriched item or None if filtered out
        """
        platform = item.get("platform")
        
        if platform in self.platform_agents:
            try:
                return self.platform_agents[platform].enrich(item)
            except Exception as e:
                logger.error(f"Error enriching {platform} item: {str(e)}")
        else:
            logger.warning(f"No agent for platform {platform} in region {self.region_id}")
        
        return None
# main.py
import logging
from agents.master_controller import MasterController
import data_collection_config 
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/promtrend.log",encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("PrompTrend")

def main():
    """Main entry point for the PrompTrend data collection system."""
    logger.info("Starting PrompTrend data collection system")
    
    # Initialize and start the master controller
    controller = MasterController()
    controller.initialize_agents()
    controller.run_collection_pipeline()
    
    logger.info("PrompTrend data collection cycle completed")

if __name__ == "__main__":
    main()
# utils/storage_manager.py
import logging
import json
import os
from datetime import datetime
import threading
from agents.discord_agent.discord_config import (

DATA_DIR,

)

logger = logging.getLogger("StorageManager")

class StorageManager:
    """
    Manages storage and retrieval of detected vulnerabilities.
    
    Stores data in JSON files for persistence and easy inspection.
    Uses a simple file-based approach for storing vulnerability data
    and tracking known channels.
    """
    
    def __init__(self, storage_dir=DATA_DIR):
        """
        Initialize the storage manager.
        
        Args:
            storage_dir: Directory to store JSON files
        """
        self.storage_dir = storage_dir
        self.vulnerabilities_file = os.path.join(storage_dir, "vulnerabilities.json")
        self.known_channels_file = os.path.join(storage_dir, "known_channels.json")
        self.lock = threading.Lock()  # For thread safety when writing files
        self._init_storage()
        
    def _init_storage(self):
        """Initialize the storage directory and files."""
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Initialize vulnerabilities file if it doesn't exist
        if not os.path.exists(self.vulnerabilities_file):
            with open(self.vulnerabilities_file, 'w') as file:
                json.dump([], file)
        
        # Initialize known channels file if it doesn't exist
        if not os.path.exists(self.known_channels_file):
            with open(self.known_channels_file, 'w') as file:
                json.dump({}, file)
        
    def store_vulnerability(self, vulnerability_data):
        """
        Store a detected vulnerability as an individual JSON file.
        """
        try:
            now = datetime.utcnow().isoformat()

            # Build a stable filename
            server = vulnerability_data.get("server_name", "unknown")
            channel = vulnerability_data.get("channel_name", "unknown")
            message_id = vulnerability_data.get("message_ids", ["unknown"])[0]
            platform = "discord"
            file_id = f"{platform}__{server}__{channel}__{message_id}".replace(" ", "_")

            # Build full path
            vuln_dir = os.path.join(self.storage_dir, "vulnerabilities")
            os.makedirs(vuln_dir, exist_ok=True)
            file_path = os.path.join(vuln_dir, f"{file_id}.json")

            # Add core metadata
            vulnerability_data["id"] = file_id
            vulnerability_data["platform"] = platform
            vulnerability_data["collected_at"] = int(datetime.utcnow().timestamp())
            vulnerability_data["created_at"] = now

            # Write to individual file
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(vulnerability_data, f, indent=2, default=str)

            logger.info(f"✅ Stored vulnerability: {file_path}")

        except Exception as e:
            logger.error(f"❌ Failed to store individual vulnerability: {e}")

    
    def is_channel_known(self, channel_id):
        """
        Check if a channel has been historically collected.
        
        Args:
            channel_id: Discord channel ID
            
        Returns:
            bool: True if channel has been historically collected
        """
        channel_id = str(channel_id)
        
        try:
            with open(self.known_channels_file, 'r') as file:
                known_channels = json.load(file)
                return channel_id in known_channels
        except (json.JSONDecodeError, FileNotFoundError):
            logger.error("Error reading known channels file")
            return False
    
    def mark_channel_known(self, channel_id):
        """
        Mark a channel as having been historically collected.
        
        Args:
            channel_id: Discord channel ID
        """
        channel_id = str(channel_id)
        now = datetime.utcnow().isoformat()
        
        with self.lock:
            try:
                # Read existing known channels
                with open(self.known_channels_file, 'r') as file:
                    try:
                        known_channels = json.load(file)
                    except json.JSONDecodeError:
                        logger.error("Error reading known channels file, initializing empty dict")
                        known_channels = {}
                
                # Update or add channel
                if channel_id in known_channels:
                    known_channels[channel_id]['last_updated'] = now
                else:
                    known_channels[channel_id] = {
                        'first_seen': now,
                        'last_updated': now
                    }
                
                # Write back to file
                with open(self.known_channels_file, 'w') as file:
                    json.dump(known_channels, file, indent=2)
                    
            except Exception as e:
                logger.error(f"Error marking channel known: {str(e)}")
    
    def get_all_vulnerabilities(self):
        """
        Retrieve all stored vulnerabilities.
        
        Returns:
            list: List of vulnerability dictionaries
        """
        try:
            with open(self.vulnerabilities_file, 'r') as file:
                return json.load(file)
        except (json.JSONDecodeError, FileNotFoundError):
            logger.error("Error reading vulnerabilities file")
            return []
    
    def get_vulnerabilities_by_server(self, server_name):
        """
        Retrieve vulnerabilities for a specific server.
        
        Args:
            server_name: Name of the Discord server
            
        Returns:
            list: List of vulnerability dictionaries for the server
        """
        vulnerabilities = self.get_all_vulnerabilities()
        return [v for v in vulnerabilities if v.get('server_name') == server_name]
    
    def close(self):
        """Clean up any resources."""
        # Nothing to do for JSON file storage
        pass
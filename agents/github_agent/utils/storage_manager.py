# utils/storage_manager.py
import os
import json
import logging
import time
from datetime import datetime
from agents.github_agent.github_config import (
    DATA_DIR,
)


# Configure logger for this module
logger = logging.getLogger(__name__)


class StorageManager:
    """
    Stores vulnerability records in JSON files.
    Also supports tracking last check times for repositories.
    """

    def __init__(self, storage_dir=DATA_DIR):
        self.storage_dir =  storage_dir
        self.vuln_dir = os.path.join(self.storage_dir, "Github_vulnerabilities")
        self.vuln_file = os.path.join(self.storage_dir, "vulnerabilities.json")
        self.meta_file = os.path.join(self.storage_dir, "metadata.json")
        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(self.vuln_dir, exist_ok=True)  
        # Load or init metadata
        if os.path.exists(self.meta_file):
            with open(self.meta_file, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

    def _serialize(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return obj

    def store_vulnerability(self, record):
        """
        Append a vulnerability record to the JSON file and store it individually too.
        """
        all_data = []
        if os.path.exists(self.vuln_file):
            with open(self.vuln_file, "r") as f:
                try:
                    all_data = json.load(f)
                except json.JSONDecodeError:
                    pass

        # Convert datetime to string recursively
        def convert(obj):
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            elif isinstance(obj, datetime):
                return obj.isoformat()
            return obj

        record_clean = convert(record)
        all_data.append(record_clean)

        # Overwrite main vulnerabilities file
        with open(self.vuln_file, "w") as f:
            json.dump(all_data, f, indent=2)

        # Also write individual file
        post_id = record.get("file_path") or record.get("issue_number") or "unknown"
        ts = int(datetime.now().timestamp() * 1000)
        out_path = os.path.join(self.vuln_dir, f"vuln_{ts}.json")
        try:
            with open(out_path, "w") as f:
                json.dump(record_clean, f, indent=2)
        except Exception as e:
            logger.error(f"❌ Failed to store vulnerability {os.path.basename(out_path)}: {str(e)}")


    def get_last_check_time(self, repo_name):
        """
        Get the last check timestamp for a repo, or None.
        """
        ts = self.metadata.get(repo_name)
        if ts:
            return datetime.fromisoformat(ts)
        return None

    def set_last_check_time(self, repo_name, dt):
        """
        Update the last check time for a repo.
        """
        self.metadata[repo_name] = dt.isoformat()
        with open(self.meta_file, "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)  

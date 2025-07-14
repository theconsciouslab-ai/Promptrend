# utils/storage_manager.py

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

logger = logging.getLogger("StorageManager")


class StorageManager:
    """
    Manages persistent storage for forum metadata, states, and vulnerability findings.
    """

    def __init__(self, storage_dir: str = "data"):
        self.storage_dir = storage_dir
        self.vuln_file = os.path.join(storage_dir, "vulnerabilities.json")
        self.state_file = os.path.join(storage_dir, "agent_state.json")

        os.makedirs(storage_dir, exist_ok=True)

    def save_agent_state(self, state: Dict[str, Any]):
        """
        Save the current agent state to file.

        Args:
            state (dict): Forum metadata, last check times, stats
        """
        def default_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return str(obj)  # fallback for any other unsupported type

        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2, default=default_serializer)
        except Exception as e:
            logger.warning(f"Failed to save agent state: {str(e)}")

    def load_agent_state(self) -> Optional[Dict[str, Any]]:
        """
        Load previously saved forum metadata and last check times.

        Returns:
            dict or None
        """
        if not os.path.exists(self.state_file):
            return None

        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
                # Convert ISO strings back to datetime
                if 'last_check_times' in state:
                    state['last_check_times'] = {
                        k: datetime.fromisoformat(v)
                        for k, v in state['last_check_times'].items()
                    }
                return state
        except Exception as e:
            logger.warning(f"Failed to load agent state: {str(e)}")
            return None

    
    def get_last_check_time(self, forum_id: str) -> Optional[datetime]:
        """
        Load the last known check time for a specific forum.

        Args:
            forum_id (str): Forum ID

        Returns:
            datetime or None
        """
        state = self.load_agent_state()
        if not state or "last_check_times" not in state:
            return None

        time_str = state["last_check_times"].get(forum_id)
        if not time_str:
            return None

        try:
            return datetime.fromisoformat(time_str)
        except Exception as e:
            logger.warning(f"Invalid timestamp format for {forum_id}: {str(e)}")
            return None

    # ✅ ADD MISSING VULNERABILITY STORAGE METHOD
    def store_vulnerability(self, vulnerability: Dict[str, Any]):
        """
        Store a detected vulnerability to the JSON file.

        Args:
            vulnerability (dict): Vulnerability data with scores and metadata
        """
        try:
            # Load existing vulnerabilities
            vulnerabilities = []
            if os.path.exists(self.vuln_file):
                with open(self.vuln_file, "r", encoding="utf-8") as f:
                    try:
                        vulnerabilities = json.load(f)
                        if not isinstance(vulnerabilities, list):
                            vulnerabilities = []
                    except json.JSONDecodeError:
                        logger.warning("Corrupted vulnerabilities file, starting fresh")
                        vulnerabilities = []

            # Add new vulnerability
            vulnerabilities.append(vulnerability)

            # Save back to file
            with open(self.vuln_file, "w", encoding="utf-8") as f:
                json.dump(vulnerabilities, f, ensure_ascii=False, indent=2, default=str)

            # ✅ Accurate score logging
            final_score = vulnerability.get("scores", {}).get("final", 0.0)
            logger.info(f"Stored vulnerability: {vulnerability.get('thread_title', 'Unknown')} (Score: {final_score:.2f})")

        except Exception as e:
            logger.error(f"Failed to store vulnerability: {str(e)}")


    def load_vulnerabilities(self) -> List[Dict[str, Any]]:
        """
        Load all stored vulnerabilities.

        Returns:
            List[Dict[str, Any]]: List of vulnerability records
        """
        if not os.path.exists(self.vuln_file):
            return []

        try:
            with open(self.vuln_file, "r", encoding="utf-8") as f:
                vulnerabilities = json.load(f)
                if isinstance(vulnerabilities, list):
                    return vulnerabilities
                else:
                    logger.warning("Vulnerabilities file contains non-list data")
                    return []
        except Exception as e:
            logger.error(f"Failed to load vulnerabilities: {str(e)}")
            return []

    def get_vulnerability_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored vulnerabilities.

        Returns:
            Dict[str, Any]: Statistics about vulnerabilities
        """
        vulnerabilities = self.load_vulnerabilities()
        
        if not vulnerabilities:
            return {"total": 0, "by_forum": {}, "by_language": {}, "score_distribution": {}}

        # Calculate statistics
        stats = {
            "total": len(vulnerabilities),
            "by_forum": {},
            "by_language": {},
            "score_distribution": {"low": 0, "medium": 0, "high": 0},
            "average_score": 0.0,
            "latest_detection": None
        }

        total_score = 0
        latest_timestamp = None

        for vuln in vulnerabilities:
            # Forum stats
            forum_name = vuln.get('forum_name', 'Unknown')
            stats["by_forum"][forum_name] = stats["by_forum"].get(forum_name, 0) + 1

            # Language stats
            language = vuln.get('language', 'unknown')
            stats["by_language"][language] = stats["by_language"].get(language, 0) + 1

            # Score distribution
            score = vuln.get('final_score', 0)
            total_score += score
            
            if score < 0.3:
                stats["score_distribution"]["low"] += 1
            elif score < 0.7:
                stats["score_distribution"]["medium"] += 1
            else:
                stats["score_distribution"]["high"] += 1

            # Latest detection
            timestamp = vuln.get('timestamp')
            if timestamp and (not latest_timestamp or timestamp > latest_timestamp):
                latest_timestamp = timestamp

        # Average score
        if len(vulnerabilities) > 0:
            stats["average_score"] = total_score / len(vulnerabilities)
        
        stats["latest_detection"] = latest_timestamp

        return stats
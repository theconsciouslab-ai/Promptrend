import os
import json
import time
import re
import logging
from pathlib import Path

logger = logging.getLogger("StorageManager")

def clean_filename(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")[:150]


class StorageManager:
    """
    Minimal StorageManager that stores vulnerabilities to JSON files locally.
    """

    def __init__(self, storage_dir="Data/vulnerabilities_collected"):
        self.storage_dir = storage_dir
        Path(self.storage_dir).mkdir(parents=True, exist_ok=True)

    def store_vulnerability(self, item):
        """
        Store a vulnerability item as a JSON file with a clean filename.
        If a file with the same platform+source+id exists, skip and log clearly.
        """
        try:
            platform = item.get("platform", "unknown")
            source = (
                item.get("repo_name")
                or item.get("subreddit")
                or item.get("community", {}).get("name")
                or "unknown"
            )
            short_id = (
                item.get("id")
                or item.get("post_id")
                or item.get("file_path")
                or item.get("issue_number")
                or "unknown"
            )

            base_name = f"{platform}_{source}_{short_id}"
            base_name = clean_filename(base_name)

            # 🔍 Check for existing file
            for fname in os.listdir(self.storage_dir):
                if fname.startswith(base_name):
                    logger.info(f"[StorageManager] 🛑 Skipped duplicate vulnerability: {base_name}")
                    return False  # Do not store again

            # ✅ No duplicate found — proceed to store
            unique_time = int(time.time())
            file_name = f"{base_name}_{unique_time}.json"
            file_path = os.path.join(self.storage_dir, file_name)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(item, f, ensure_ascii=False, indent=2)

            logger.info(f"[StorageManager] ✅ Stored vulnerability: {file_path}")
            return True

        except Exception as e:
            logger.error(f"[StorageManager] ❌ Error storing vulnerability: {str(e)}")
            return False

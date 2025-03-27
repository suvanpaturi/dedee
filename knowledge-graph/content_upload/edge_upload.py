#!/usr/bin/env python3
"""
Script to upload knowledge from edge devices to central Neo4j database.
This script can be run on a schedule or triggered by events.
"""

import os
import json
import argparse
import logging
import requests
from datetime import datetime
from typing import List, Dict, Optional, Any
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("edge_upload.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Default settings
DEFAULT_API_URL = "http://localhost:5001"
DEFAULT_DEVICE_ID = os.environ.get("DEVICE_ID", f"edge-device-{os.getpid()}")
DEFAULT_SYNC_INTERVAL = 3600  # 1 hour
DEFAULT_LOCAL_CACHE = "./edge_cache.json"
DEFAULT_CENTRAL_ENDPOINT = "/update/"

def load_local_cache(cache_file: str) -> Dict[str, Any]:
    """Load the local cache of items to sync."""
    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)
        return {"last_sync": None, "pending_items": [], "sync_history": []}
    except Exception as e:
        logger.error(f"Error loading cache: {str(e)}")
        return {"last_sync": None, "pending_items": [], "sync_history": []}

def save_local_cache(cache_file: str, cache_data: Dict[str, Any]):
    """Save the local cache of items to sync."""
    try:
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving cache: {str(e)}")

def add_to_sync_queue(cache_file: str, items: List[Dict[str, Any]]):
    """Add items to the sync queue."""
    #!/usr/bin/env python3
"""
Script to upload knowledge from edge devices to central Neo4j database.
This script can be run on a schedule or triggered by events.
"""

import os
import json
import argparse
import logging
import requests
from datetime import datetime
from typing import List, Dict, Optional, Any
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("edge_upload.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Default settings
DEFAULT_API_URL = "http://localhost:5001"
DEFAULT_DEVICE_ID = os.environ.get("DEVICE_ID", f"edge-device-{os.getpid()}")
DEFAULT_SYNC_INTERVAL = 3600  # 1 hour
DEFAULT_LOCAL_CACHE = "./edge_cache.json"
DEFAULT_CENTRAL_ENDPOINT = "/update/"

def load_local_cache(cache_file: str) -> Dict[str, Any]:
    """Load the local cache of items to sync."""
    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)
        return {"last_sync": None, "pending_items": [], "sync_history": []}
    except Exception as e:
        logger.error(f"Error loading cache: {str(e)}")
        return {"last_sync": None, "pending_items": [], "sync_history": []}

def save_local_cache(cache_file: str, cache_data: Dict[str, Any]):
    """Save the local cache of items to sync."""
    try:
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving cache: {str(e)}")

def add_to_sync_queue(cache_file: str, items: List[Dict[str, Any]]):
    """Add items to the sync queue."""
    cache_data = load_local_cache(cache_file)
    cache_data["pending_items"].extend(items)
    save_local_cache(cache_file, cache_data)
    logger.info(f"Added {len(items)} items to sync queue")

def sync_to_central(api_url: str, device_id: str, items: List[Dict[str, Any]]) -> bool:
    """
    Sync items to the central server.
    
    Args:
        api_url: URL of central API
        device_id: ID of this edge device
        items: List of items to sync
        
    Returns:
        True if sync was successful, False otherwise
    """
    if not items:
        logger.info("No items to sync")
        return True
    
    try:
        # Prepare the data with device ID
        payload = {
            "items": items
        }
        
        # Add device ID in headers
        headers = {
            "Content-Type": "application/json",
            "X-Device-ID": device_id
        }
        
        # Send the data to the central server
        url = f"{api_url}{DEFAULT_CENTRAL_ENDPOINT}"
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            logger.info(f"Successfully synced {len(items)} items to central server")
            return True
        else:
            logger.error(f"Failed to sync: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error syncing to central: {str(e)}")
        return False

def process_sync_queue(cache_file: str, api_url: str, device_id: str):
    """Process the sync queue."""
    cache_data = load_local_cache(cache_file)
    pending_items = cache_data["pending_items"]
    
    if not pending_items:
        logger.info("No pending items to sync")
        return
    
    logger.info(f"Attempting to sync {len(pending_items)} items")
    
    # Try to sync items
    success = sync_to_central(api_url, device_id, pending_items)
    
    if success:
        # Update sync history
        timestamp = datetime.now().isoformat()
        cache_data["last_sync"] = timestamp
        cache_data["sync_history"].append({
            "timestamp": timestamp,
            "count": len(pending_items),
            "status": "success"
        })
        
        # Clear pending items
        cache_data["pending_items"] = []
        
        # Limit sync history size
        if len(cache_data["sync_history"]) > 100:
            cache_data["sync_history"] = cache_data["sync_history"][-100:]
    else:
        # Record failed sync attempt
        cache_data["sync_history"].append({
            "timestamp": datetime.now().isoformat(),
            "count": len(pending_items),
            "status": "failed"
        })
    
    save_local_cache(cache_file, cache_data)

def collect_new_knowledge(source_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Collect new knowledge items from the edge device.
    This could be from local logs, a database, or user input.
    
    Args:
        source_file: Optional file to read items from
        
    Returns:
        List of knowledge items
    """
    items = []
    
    if source_file and os.path.exists(source_file):
        try:
            with open(source_file, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict) and "items" in data:
                items = data["items"]
        except Exception as e:
            logger.error(f"Error reading source file: {str(e)}")
    
    return items

def run_continuous_sync(api_url: str, device_id: str, cache_file: str, 
                       interval: int, source_file: Optional[str] = None):
    """
    Run continuous sync process.
    
    Args:
        api_url: URL of central API
        device_id: ID of this edge device
        cache_file: Path to local cache file
        interval: Sync interval in seconds
        source_file: Optional file to read items from
    """
    logger.info(f"Starting continuous sync for device {device_id}")
    logger.info(f"API URL: {api_url}")
    logger.info(f"Sync interval: {interval} seconds")
    logger.info(f"Cache file: {cache_file}")
    
    try:
        while True:
            # Collect new knowledge
            if source_file:
                new_items = collect_new_knowledge(source_file)
                if new_items:
                    add_to_sync_queue(cache_file, new_items)
            
            # Process sync queue
            process_sync_queue(cache_file, api_url, device_id)
            
            # Wait for next sync
            logger.info(f"Sleeping for {interval} seconds until next sync")
            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("Sync process interrupted by user")
    except Exception as e:
        logger.error(f"Error in sync process: {str(e)}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Edge device knowledge upload script")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="URL of central API")
    parser.add_argument("--device-id", default=DEFAULT_DEVICE_ID, help="ID of this edge device")
    parser.add_argument("--cache-file", default=DEFAULT_LOCAL_CACHE, help="Path to local cache file")
    parser.add_argument("--interval", type=int, default=DEFAULT_SYNC_INTERVAL, help="Sync interval in seconds")
    parser.add_argument("--source", help="Source file with knowledge items")
    parser.add_argument("--once", action="store_true", help="Run sync once and exit")
    
    args = parser.parse_args()
    
    # If source is provided but once is not, add items to queue and exit
    if args.source and not args.once:
        new_items = collect_new_knowledge(args.source)
        if new_items:
            add_to_sync_queue(args.cache_file, new_items)
            logger.info(f"Added {len(new_items)} items to sync queue")
        return
    
    # If once is provided, run sync once and exit
    if args.once:
        if args.source:
            new_items = collect_new_knowledge(args.source)
            if new_items:
                add_to_sync_queue(args.cache_file, new_items)
        process_sync_queue(args.cache_file, args.api_url, args.device_id)
        return
    
    # Otherwise, run continuous sync
    run_continuous_sync(args.api_url, args.device_id, args.cache_file, 
                      args.interval, args.source)

if __name__ == "__main__":
    main()
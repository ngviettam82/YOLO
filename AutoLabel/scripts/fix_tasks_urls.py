#!/usr/bin/env python3
"""
Fix tasks.json URLs to use http://localhost:8000/images/ prefix
Run this after import_to_label_studio.py if images don't load in Label Studio
"""

import json
from pathlib import Path
import sys

def fix_tasks_urls(project_dir, port=8000):
    """Update all image paths in tasks.json to use HTTP server"""
    
    tasks_file = Path(project_dir) / "tasks.json"
    
    if not tasks_file.exists():
        print(f"❌ Error: tasks.json not found in {project_dir}")
        return False
    
    print(f"📂 Loading tasks.json from: {project_dir}")
    
    # Read tasks.json
    with open(tasks_file, 'r') as f:
        tasks = json.load(f)
    
    # Fix URLs
    updated = 0
    for task in tasks:
        if "data" in task and "image" in task["data"]:
            old_image = task["data"]["image"]
            
            # Extract filename
            if isinstance(old_image, str):
                filename = Path(old_image).name
                new_url = f"http://localhost:{port}/images/{filename}"
                
                if old_image != new_url:
                    print(f"  • {filename}")
                    task["data"]["image"] = new_url
                    updated += 1
    
    if updated == 0:
        print(f"ℹ️  All URLs already correct!")
        return True
    
    # Write back
    with open(tasks_file, 'w') as f:
        json.dump(tasks, f, indent=2)
    
    print(f"\n✅ Updated {updated} URLs")
    print(f"✅ tasks.json saved\n")
    print(f"Next steps:")
    print(f"1. Start image server: python serve_images.py")
    print(f"2. Start Label Studio: label-studio")
    print(f"3. Import tasks.json into Label Studio")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_tasks_urls.py <project_directory> [port]")
        print("\nExample:")
        print("  python fix_tasks_urls.py label_studio_MyProject")
        print("  python fix_tasks_urls.py label_studio_MyProject 8001")
        sys.exit(1)
    
    project_dir = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    
    if not fix_tasks_urls(project_dir, port):
        sys.exit(1)

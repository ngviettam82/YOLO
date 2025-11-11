#!/usr/bin/env python3
"""
Label Studio Integration Tool
Import auto-labeled YOLO annotations into Label Studio for review and adjustment
"""

import sys
from pathlib import Path
import json
import shutil
import subprocess
import time
import webbrowser
from typing import List, Dict, Tuple
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import threading
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))


class LabelStudioImporter:
    def __init__(self):
        self.image_folder = None
        self.label_folder = None
        self.project_name = None
        self.project_dir = None
        self.class_names = {}  # Load from data.yaml
        self.load_class_names()
        
    def load_class_names(self):
        """Load class names from dataset config"""
        data_yaml = Path(PROJECT_ROOT) / "dataset" / "data.yaml"
        if data_yaml.exists():
            try:
                with open(data_yaml, 'r') as f:
                    data = yaml.safe_load(f)
                    if data and 'names' in data:
                        self.class_names = data['names']
                        print(f"📋 Loaded {len(self.class_names)} class names from data.yaml")
            except Exception as e:
                print(f"⚠️  Could not load class names from data.yaml: {e}")
                # Use default numeric class names
                self.class_names = {i: f"class_{i}" for i in range(10)}
        else:
            # Use default numeric class names
            self.class_names = {i: f"class_{i}" for i in range(10)}
        
    def select_folders(self):
        """Select image and label folders"""
        root = tk.Tk()
        root.withdraw()
        
        # Select image folder
        initial_dir = Path(PROJECT_ROOT) / "dataset" / "images"
        if not initial_dir.exists():
            initial_dir = PROJECT_ROOT
        
        image_folder = filedialog.askdirectory(
            title="Select Image Folder",
            initialdir=str(initial_dir)
        )
        
        if not image_folder:
            root.destroy()
            return False
        
        self.image_folder = Path(image_folder)
        
        # Select label folder
        initial_dir = Path(PROJECT_ROOT) / "dataset" / "labels"
        if not initial_dir.exists():
            initial_dir = PROJECT_ROOT
        
        label_folder = filedialog.askdirectory(
            title="Select Label Folder (YOLO format .txt files)",
            initialdir=str(initial_dir)
        )
        
        root.destroy()
        
        if not label_folder:
            return False
        
        self.label_folder = Path(label_folder)
        return True
    
    def get_project_name(self):
        """Get project name from user"""
        root = tk.Tk()
        root.withdraw()
        
        project_name = tk.simpledialog.askstring(
            "Project Name",
            "Enter Label Studio project name:\n(e.g., 'My Dataset Review')",
            parent=root,
            initialvalue=f"YOLO_Review_{self.image_folder.name}"
        )
        
        root.destroy()
        
        if not project_name:
            return False
        
        self.project_name = project_name
        self.project_dir = Path(PROJECT_ROOT) / f"label_studio_{project_name.replace(' ', '_')}"
        return True
    
    def read_yolo_labels(self, label_file: Path, image_width: int, image_height: int) -> List[Dict]:
        """Convert YOLO format labels to Label Studio format"""
        tasks = []
        
        if not label_file.exists() or label_file.stat().st_size == 0:
            return tasks
        
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Convert normalized to pixel coordinates
                        x_left = (x_center - width / 2) * image_width
                        y_top = (y_center - height / 2) * image_height
                        box_width = width * image_width
                        box_height = height * image_height
                        
                        # Clamp values
                        x_left = max(0, min(x_left, image_width))
                        y_top = max(0, min(y_top, image_height))
                        box_width = max(0, min(box_width, image_width - x_left))
                        box_height = max(0, min(box_height, image_height - y_top))
                        
                        # Get class name from loaded names, default to numeric string
                        class_name = self.class_names.get(class_id, f"class_{class_id}")
                        
                        tasks.append({
                            "value": {
                                "x": (x_left / image_width) * 100,
                                "y": (y_top / image_height) * 100,
                                "width": (box_width / image_width) * 100,
                                "height": (box_height / image_height) * 100,
                                "rotation": 0,
                                "rectanglelabels": [str(class_name)]
                            },
                            "from_name": "label",
                            "to_name": "image",
                            "type": "rectanglelabels"
                        })
        except Exception as e:
            print(f"Error reading label file {label_file}: {e}")
        
        return tasks
    
    def get_image_dimensions(self, image_file: Path) -> Tuple[int, int]:
        """Get image dimensions"""
        try:
            from PIL import Image
            with Image.open(image_file) as img:
                return img.width, img.height
        except Exception as e:
            print(f"Error getting image dimensions: {e}")
            return 640, 480
    
    def create_label_studio_config(self) -> str:
        """Create Label Studio XML configuration using actual class names"""
        # Define colors for different classes
        colors = ["green", "blue", "red", "yellow", "purple", "orange", "cyan", "magenta", "lime", "pink"]
        
        # Build Label elements with actual class names
        labels_xml = ""
        for class_id, class_name in sorted(self.class_names.items()):
            color = colors[class_id % len(colors)]
            labels_xml += f'    <Label value="{class_name}" background="{color}"/>\n'
        
        config = f"""<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
{labels_xml}  </RectangleLabels>
</View>"""
        return config
    
    def show_config_popup(self, config: str):
        """Show XML configuration in a separate popup window for copy-paste"""
        # Create hidden root window
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        
        popup = tk.Toplevel(root)
        popup.title("📋 Label Studio Configuration - Copy This!")
        popup.geometry("900x600")
        
        # Header
        header = tk.Label(popup, text="Copy this XML into Label Studio → Project Settings → Labeling Interface", 
                         font=("Arial", 12, "bold"), fg="blue", bg="lightyellow", pady=10)
        header.pack(fill=tk.X)
        
        # Instructions
        instructions = tk.Label(popup, text="""
Steps:
1. Copy the XML below (Ctrl+A to select all, Ctrl+C to copy)
2. Go to Label Studio → Project Settings → Labeling Interface
3. Replace the default XML with this code
4. Click 'Save'
5. Import your tasks.json
        """, justify=tk.LEFT, font=("Arial", 10))
        instructions.pack(fill=tk.X, padx=10, pady=10)
        
        # Text area for config
        text_frame = tk.Frame(popup)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text_widget = scrolledtext.ScrolledText(text_frame, height=20, width=100, 
                                               yscrollcommand=scrollbar.set, font=("Courier", 10))
        text_widget.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=text_widget.yview)
        
        text_widget.insert(tk.END, config)
        text_widget.config(state=tk.DISABLED)  # Read-only
        
        # Copy button
        def copy_to_clipboard():
            popup.clipboard_clear()
            popup.clipboard_append(config)
            messagebox.showinfo("Success", "XML configuration copied to clipboard!\n\nNow paste it into Label Studio.")
        
        button_frame = tk.Frame(popup)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        copy_btn = tk.Button(button_frame, text="📋 Copy to Clipboard", command=copy_to_clipboard, 
                            bg="lightgreen", font=("Arial", 11), padx=20, pady=5)
        copy_btn.pack(side=tk.LEFT, padx=5)
        
        close_btn = tk.Button(button_frame, text="Close", command=popup.destroy, 
                             font=("Arial", 11), padx=20, pady=5)
        close_btn.pack(side=tk.LEFT, padx=5)
        
        def on_popup_close():
            popup.destroy()
            root.destroy()
        
        popup.protocol("WM_DELETE_WINDOW", on_popup_close)
        popup.wait_window()
    
    def start_image_server_background(self):
        """Start the image server in a background thread"""
        def run_server():
            try:
                import http.server
                import socketserver
                from urllib.parse import unquote
                
                PORT = 8000
                IMAGES_DIR = self.project_dir / "images"
                
                class ImageHandler(http.server.SimpleHTTPRequestHandler):
                    def do_GET(self):
                        if self.path.startswith("/images/"):
                            filename = unquote(self.path[8:])
                            filepath = IMAGES_DIR / filename
                            
                            if filepath.exists() and filepath.is_file():
                                try:
                                    with open(filepath, 'rb') as f:
                                        self.send_response(200)
                                        self.send_header('Content-type', 'image/jpeg')
                                        self.send_header('Content-Length', filepath.stat().st_size)
                                        self.end_headers()
                                        self.wfile.write(f.read())
                                except Exception as e:
                                    self.send_response(500)
                                    self.end_headers()
                            else:
                                self.send_response(404)
                                self.end_headers()
                        else:
                            self.send_response(404)
                            self.end_headers()
                
                with socketserver.TCPServer(("", PORT), ImageHandler) as httpd:
                    print(f"\n✅ Image server started on http://localhost:{PORT}")
                    print(f"📁 Serving images from: {IMAGES_DIR}")
                    print(f"⚠️  Keep this window open while using Label Studio!\n")
                    httpd.serve_forever()
            except Exception as e:
                print(f"❌ Error starting image server: {e}")
        
        # Start server in background thread
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
    
    def create_label_studio_project(self) -> bool:
        """Create Label Studio project with auto-labeled data"""
        try:
            print(f"\n{'='*80}")
            print(f"📁 Creating Label Studio Project")
            print(f"{'='*80}\n")
            
            # Create project directory
            self.project_dir.mkdir(parents=True, exist_ok=True)
            print(f"📂 Project directory: {self.project_dir}\n")
            
            # Copy images to project
            images_dir = self.project_dir / "images"
            images_dir.mkdir(exist_ok=True)
            
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
            image_files = [f for f in self.image_folder.iterdir() 
                          if f.suffix.lower() in image_extensions]
            
            print(f"📋 Found {len(image_files)} images")
            print(f"📋 Copying images to project...\n")
            
            for image_file in image_files:
                dest = images_dir / image_file.name
                shutil.copy2(image_file, dest)
            
            # Create tasks JSON
            tasks = []
            task_id = 1  # Start ID from 1
            annotation_id = 1  # Start annotation ID from 1
            
            for image_file in image_files:
                label_file = self.label_folder / f"{image_file.stem}.txt"
                img_width, img_height = self.get_image_dimensions(image_file)
                
                annotations = self.read_yolo_labels(label_file, img_width, img_height)
                
                # Use HTTP URL pointing to image server (default: localhost:8000)
                task = {
                    "id": task_id,  # Unique task ID
                    "data": {
                        "image": f"http://localhost:8000/images/{image_file.name}"
                    }
                }
                
                if annotations:
                    task["annotations"] = [{
                        "id": annotation_id,  # Unique annotation ID
                        "completed_by": 1,
                        "result": annotations,
                        "was_cancelled": False,
                        "ground_truth": True
                    }]
                    annotation_id += 1  # Increment for next annotation
                
                tasks.append(task)
                task_id += 1  # Increment for next task
            
            # Write tasks.json
            tasks_file = self.project_dir / "tasks.json"
            with open(tasks_file, 'w', encoding='utf-8') as f:
                json.dump(tasks, f, indent=2)
            
            print(f"✓ Created tasks.json with {len(tasks)} tasks\n")
            
            # Create label config
            config = self.create_label_studio_config()
            config_file = self.project_dir / "label_config.xml"
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config)
            
            print(f"✓ Created label_config.xml\n")
            
            # Show XML config in popup window
            self.show_config_popup(config)
            
            # Start image server in background
            self.start_image_server_background()
            
            # Create README
            readme = f"""# Label Studio Project: {self.project_name}

## ⚠️ IMPORTANT: Image Server Setup

Label Studio on Windows needs an HTTP server to access images properly.

## ⚠️ IMPORTANT: Label Configuration Setup

Before importing your labels, you MUST set the labeling interface:

1. Open Label Studio → Create New Project
2. Go to Project Settings → Labeling Interface
3. **Replace the default XML with the content from `label_config.xml` in this folder**
4. Click "Save"
5. Now import `tasks.json` and your labels will be recognized!

The `label_config.xml` contains your label definitions with these classes:
"""
            # Add class names to README
            for class_id, class_name in sorted(self.class_names.items()):
                readme += f"- {class_name}\n"
            
            readme += f"""
### Quick Setup (3 steps):

**Step 1: Get Label Configuration**
- Copy the XML that appeared in the popup window
- (Or open `label_config.xml` in this folder)

**Step 2: Set Label Config in Label Studio**
- Go to Label Studio → Create Project
- Go to Settings → Labeling Interface
- Paste the XML configuration
- Click "Save"

**Step 3: Start Label Studio**
```powershell
.venv\\Scripts\\activate.ps1
label-studio
```

✅ **Image server is already running automatically!**

Then:
1. Open Label Studio at `http://localhost:8080`
2. Import `tasks.json` from this directory
3. Review and edit labels
4. Export when done

## Project Details

- Images: {len(image_files)} images
- Tasks: {len(tasks)} tasks
- Classes: {", ".join(self.class_names.values())}
- Format: YOLO with auto-generated labels
- Image Server: Running on localhost:8000 (automatic)

## Files in This Project

- `tasks.json` - Task definitions with image references
- `label_config.xml` - **COPY THIS INTO LABEL STUDIO SETTINGS** ← Important!
- `images/` - Folder containing all images
- `README.md` - This file

## Troubleshooting

**"Labeling box has no label" error in Label Studio?**
- You didn't set the label configuration!
- Copy `label_config.xml` into Project Settings → Labeling Interface

**Images won't load?**
- Make sure the image server is still running (the window should show "Image server started on http://localhost:8000")
- Check that `tasks.json` has correct URLs: `http://localhost:8000/images/...`
- Check browser console for errors (F12)

**"Cannot find module"?**
- Make sure you're in the virtual environment: `.venv\\Scripts\\activate.ps1`

**Port already in use?**
- Another application is using port 8000
- Check if port 8000 is already in use and close that application

## Workflow

1. Set label configuration (label_config.xml)
2. Review labels in Label Studio
3. Make corrections (move, resize, delete boxes)
4. Add missing boxes
5. Click Submit when done
6. Export annotations
7. Use for training

"""
            readme_file = self.project_dir / "README.md"
            with open(readme_file, 'w', encoding='utf-8') as f:
                f.write(readme)
            
            print(f"✓ Created README.md")
            print(f"✓ Image server running on http://localhost:8000\n")
            print(f"{'='*80}")
            print(f"✅ Label Studio project created successfully!")
            print(f"{'='*80}\n")
            print(f"Project location: {self.project_dir}\n")
            print(f"📋 Class labels that will be used:")
            for class_id, class_name in sorted(self.class_names.items()):
                print(f"   • {class_name}")
            print(f"\n🚀 Next steps:")
            print(f"1. Copy the XML configuration from the popup window (or from label_config.xml)")
            print(f"2. Open Label Studio → Create Project")
            print(f"3. Go to Project Settings → Labeling Interface")
            print(f"4. Paste the XML configuration and click 'Save'")
            print(f"5. Import tasks.json from: {self.project_dir}")
            print(f"6. Image server is running on http://localhost:8000")
            print(f"7. Review and adjust labels")
            print(f"8. Export annotations\n")
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating Label Studio project: {str(e)}")
            messagebox.showerror("Error", f"Failed to create project:\n{str(e)}")
            return False
    
    def launch_label_studio_guide(self):
        """Show Label Studio usage guide"""
        guide = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                    Label Studio - Usage Guide                             ║
╚════════════════════════════════════════════════════════════════════════════╝

PROJECT CREATED! ✓

Your auto-labeled data has been prepared for Label Studio.

════════════════════════════════════════════════════════════════════════════

⚡ QUICK START (3 STEPS):

TERMINAL 1 - Start Image Server:
   cd {self.project_dir}
   python serve_images.py
   
   You should see: "Serving images on http://localhost:8000"

TERMINAL 2 - Start Label Studio:
   .venv\\Scripts\\activate.ps1
   label-studio
   
   Wait for browser to open at http://localhost:8080

BROWSER - Review Labels:
   1. Create new project (name: "Review")
   2. Import → Select tasks.json from project folder
   3. Click on images and edit boxes
   4. Export when done

════════════════════════════════════════════════════════════════════════════

📋 DETAILED STEPS:

STEP 1: Start Image Server (Terminal 1)
   ├─ Open PowerShell
   ├─ cd {self.project_dir}
   ├─ python serve_images.py
   └─ Wait for "Serving images on http://localhost:8000"
   
   ⚠️ KEEP THIS RUNNING! Don't close this terminal!

STEP 2: Start Label Studio (Terminal 2 - NEW window)
   ├─ Open NEW PowerShell
   ├─ cd C:\\Users\\ADMIN\\Documents\\Code\\YOLO
   ├─ .venv\\Scripts\\activate.ps1
   ├─ label-studio
   └─ Wait for browser to open

STEP 3: Import Project (Browser)
   ├─ Label Studio opens at http://localhost:8080
   ├─ Click "Create Project"
   ├─ Name: "Review_Labels"
   ├─ Language: English
   ├─ Click "Create"
   ├─ Click "Start"
   ├─ Click "Import"
   ├─ Select: tasks.json from {self.project_dir}
   ├─ Import settings will auto-detect images
   └─ Click "Import"

STEP 4: Review and Edit Labels (Browser)
   ├─ Images will load from http://localhost:8000
   ├─ Boxes will appear on images
   ├─ Edit boxes:
   │  ├─ Drag to move
   │  ├─ Drag corners to resize
   │  ├─ Right-click to delete
   │  └─ Use Rectangle tool to add
   ├─ Change labels if needed
   └─ Click "Submit" when done

STEP 5: Export (Browser)
   ├─ Click "Export"
   ├─ Select format (YOLO if available)
   └─ Save file for training

════════════════════════════════════════════════════════════════════════════

🎯 EDITING TIPS:

Moving a Box:
   • Click inside the box
   • Drag to new position
   • Release to drop

Resizing a Box:
   • Hover over corner
   • Drag corner outward/inward
   • Release to resize

Deleting a Box:
   • Right-click on box
   • Select "Delete"

Adding a Box:
   • Click "Rectangle" tool in toolbar
   • Draw box on image
   • Select class label
   • Release

Changing Class:
   • Click on labeled box
   • Select new class from dropdown

════════════════════════════════════════════════════════════════════════════

⚠️ TROUBLESHOOTING:

Problem: "There was an issue loading URL"
Solution: Make sure serve_images.py is running in Terminal 1

Problem: "Cannot find image file"
Solution: 
   1. Check serve_images.py is running
   2. Check tasks.json has "image": "http://localhost:8000/images/..."

Problem: "Port already in use"
Solution: 
   1. Close serve_images.py
   2. Edit serve_images.py, change PORT = 8000 to PORT = 8001
   3. Run again

Problem: Images still won't load
Solution:
   1. Open browser console (F12)
   2. Look for error messages
   3. Check that http://localhost:8000/images/test.jpg loads in browser
   4. Try clearing browser cache (Ctrl+Shift+Delete)

════════════════════════════════════════════════════════════════════════════

📂 PROJECT FILES:

Location: {self.project_dir}

Files:
   • tasks.json        - Task definitions (auto-generated)
   • label_config.xml  - Annotation schema
   • serve_images.py   - Image server (run this first!)
   • images/           - All your images
   • README.md         - Full documentation

════════════════════════════════════════════════════════════════════════════

✅ Ready to start! Follow the QUICK START steps above.

"""
        return guide
    
    def run(self):
        """Main execution flow"""
        print(f"\n{'='*80}")
        print(f"🏷️  Label Studio Importer")
        print(f"{'='*80}\n")
        
        # Step 1: Select folders
        if not self.select_folders():
            print("❌ Operation cancelled")
            return
        
        print(f"✓ Image folder: {self.image_folder}")
        print(f"✓ Label folder: {self.label_folder}\n")
        
        # Step 2: Get project name
        if not self.get_project_name():
            print("❌ Operation cancelled")
            return
        
        print(f"✓ Project name: {self.project_name}\n")
        
        # Step 3: Create project
        if not self.create_label_studio_project():
            return
        
        # Step 4: Show guide (print only, no popup window)
        guide = self.launch_label_studio_guide()
        print(guide)


def main():
    """Main entry point"""
    try:
        importer = LabelStudioImporter()
        importer.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    main()

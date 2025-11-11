#!/usr/bin/env python3
"""
Label Verification Tool
Visualizes generated YOLO format labels on images for verification
"""

import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import threading

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))


class LabelVerifier:
    def __init__(self):
        self.image_folder = None
        self.label_folder = None
        self.image_files = []
        self.current_index = 0
        self.window = None
        self.canvas = None
        self.label_text = None
        
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
            title="Select Label Folder",
            initialdir=str(initial_dir)
        )
        
        root.destroy()
        
        if not label_folder:
            return False
        
        self.label_folder = Path(label_folder)
        return True
    
    def get_image_files(self):
        """Get all image files"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
        image_files = [
            f for f in self.image_folder.iterdir()
            if f.suffix.lower() in image_extensions
        ]
        return sorted(image_files)
    
    def read_labels(self, image_file):
        """Read label file for image"""
        label_file = self.label_folder / f"{image_file.stem}.txt"
        
        if not label_file.exists():
            return []
        
        labels = []
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
                        labels.append((class_id, x_center, y_center, width, height))
        except Exception as e:
            print(f"Error reading label file: {e}")
        
        return labels
    
    def draw_labels(self, image, labels, image_height, image_width):
        """Draw bounding boxes on image"""
        # Define colors for different classes (BGR format for OpenCV)
        colors = [
            (0, 255, 0),      # Green - Class 0
            (255, 0, 0),      # Blue - Class 1
            (0, 0, 255),      # Red - Class 2
            (255, 255, 0),    # Cyan - Class 3
            (255, 0, 255),    # Magenta - Class 4
            (0, 255, 255),    # Yellow - Class 5
        ]
        
        for class_id, x_center, y_center, width, height in labels:
            # Convert from normalized to pixel coordinates
            x_center_px = int(x_center * image_width)
            y_center_px = int(y_center * image_height)
            width_px = int(width * image_width)
            height_px = int(height * image_height)
            
            # Calculate top-left and bottom-right corners
            x1 = int(x_center_px - width_px / 2)
            y1 = int(y_center_px - height_px / 2)
            x2 = int(x_center_px + width_px / 2)
            y2 = int(y_center_px + height_px / 2)
            
            # Clamp to image boundaries
            x1 = max(0, min(x1, image_width - 1))
            y1 = max(0, min(y1, image_height - 1))
            x2 = max(0, min(x2, image_width))
            y2 = max(0, min(y2, image_height))
            
            # Draw rectangle
            color = colors[class_id % len(colors)]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw class label
            text = f"Class {class_id}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, text, (x1, y1 - 5), font, 0.5, color, 2)
        
        return image
    
    def show_image(self, index):
        """Display image at index"""
        if not 0 <= index < len(self.image_files):
            return
        
        self.current_index = index
        image_file = self.image_files[index]
        
        # Read image
        image = cv2.imread(str(image_file))
        if image is None:
            messagebox.showerror("Error", f"Cannot read image: {image_file}")
            return
        
        image_height, image_width = image.shape[:2]
        
        # Read labels
        labels = self.read_labels(image_file)
        
        # Draw labels
        image_with_boxes = self.draw_labels(image.copy(), labels, image_height, image_width)
        
        # Convert BGR to RGB for display
        image_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
        
        # Resize for display if too large
        max_size = 800
        if max(image_height, image_width) > max_size:
            scale = max_size / max(image_height, image_width)
            new_width = int(image_width * scale)
            new_height = int(image_height * scale)
            image_rgb = cv2.resize(image_rgb, (new_width, new_height))
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        tk_image = ImageTk.PhotoImage(pil_image)
        
        # Update canvas
        self.canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
        self.canvas.image = tk_image  # Keep a reference
        
        # Update label text
        has_labels = "✓" if labels else "✗"
        status = f"Image {self.current_index + 1}/{len(self.image_files)} | {image_file.name} | Labels: {has_labels} ({len(labels)} detections)"
        self.label_text.set(status)
    
    def next_image(self):
        """Show next image"""
        self.show_image(self.current_index + 1)
    
    def prev_image(self):
        """Show previous image"""
        self.show_image(self.current_index - 1)
    
    def run(self):
        """Run the label verifier"""
        print("\n" + "="*80)
        print("🔍 Label Verification Tool")
        print("="*80 + "\n")
        
        # Select folders
        if not self.select_folders():
            print("❌ No folders selected")
            return
        
        print(f"📁 Image folder: {self.image_folder}")
        print(f"📁 Label folder: {self.label_folder}")
        
        # Get image files
        self.image_files = self.get_image_files()
        
        if not self.image_files:
            messagebox.showerror("Error", "No image files found")
            print("❌ No image files found")
            return
        
        print(f"📊 Found {len(self.image_files)} images\n")
        
        # Create GUI window
        self.window = tk.Tk()
        self.window.title("Label Verification Tool")
        self.window.geometry("900x700")
        
        # Canvas for image
        self.canvas = tk.Canvas(self.window, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Label text
        self.label_text = tk.StringVar()
        status_label = tk.Label(self.window, textvariable=self.label_text, fg="blue")
        status_label.pack(fill=tk.X, padx=10)
        
        # Buttons
        button_frame = tk.Frame(self.window)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(button_frame, text="◀ Previous", command=self.prev_image, width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Next ▶", command=self.next_image, width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Exit", command=self.window.quit, width=15).pack(side=tk.RIGHT, padx=5)
        
        # Show first image
        self.show_image(0)
        
        print("🎯 Verification window opened. Use Previous/Next buttons to navigate.")
        print("   Green = Detected objects with labels")
        print("   ✓ = Image has labels, ✗ = Image has no labels\n")
        
        self.window.mainloop()


def main():
    """Main entry point"""
    try:
        verifier = LabelVerifier()
        verifier.run()
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        messagebox.showerror("Error", f"An error occurred:\n{str(e)}")


if __name__ == "__main__":
    main()

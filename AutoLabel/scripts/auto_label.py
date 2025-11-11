#!/usr/bin/env python3
"""
Auto-Labeling Tool using Pretrained YOLO Model
Provides GUI windows to select model and image folder, then automatically generates labels
"""

import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
from ultralytics import YOLO
import torch
import cv2
from tqdm import tqdm
import time

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))


class AutoLabelGUI:
    def __init__(self):
        self.model_path = None
        self.image_folder = None
        self.output_folder = None
        self.model = None
        
    def select_model(self):
        """Open file dialog to select YOLO model"""
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        
        initial_dir = Path(PROJECT_ROOT) / "models"
        if not initial_dir.exists():
            initial_dir = PROJECT_ROOT
        
        file_path = filedialog.askopenfilename(
            title="Select YOLO Model",
            initialdir=str(initial_dir),
            filetypes=[
                ("PyTorch Models", "*.pt"),
                ("All Files", "*.*")
            ]
        )
        
        root.destroy()
        
        if file_path:
            self.model_path = Path(file_path)
            print(f"✅ Model selected: {self.model_path.name}")
            return True
        else:
            print("❌ No model selected")
            return False
    
    def select_image_folder(self):
        """Open folder dialog to select image folder"""
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        
        initial_dir = Path(PROJECT_ROOT) / "dataset" / "images"
        if not initial_dir.exists():
            initial_dir = PROJECT_ROOT
        
        folder_path = filedialog.askdirectory(
            title="Select Image Folder",
            initialdir=str(initial_dir)
        )
        
        root.destroy()
        
        if folder_path:
            self.image_folder = Path(folder_path)
            print(f"✅ Image folder selected: {self.image_folder}")
            return True
        else:
            print("❌ No folder selected")
            return False
    
    def select_output_folder(self):
        """Open folder dialog to select output folder for labels"""
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        
        initial_dir = Path(PROJECT_ROOT) / "dataset" / "labels"
        if not initial_dir.exists():
            initial_dir = Path(PROJECT_ROOT) / "dataset"
        
        folder_path = filedialog.askdirectory(
            title="Select Output Folder for Labels",
            initialdir=str(initial_dir)
        )
        
        root.destroy()
        
        if folder_path:
            self.output_folder = Path(folder_path)
            self.output_folder.mkdir(parents=True, exist_ok=True)
            print(f"✅ Output folder selected: {self.output_folder}")
            return True
        else:
            print("❌ No output folder selected")
            return False
    
    def load_model(self):
        """Load YOLO model"""
        try:
            print(f"\n📦 Loading model: {self.model_path.name}")
            self.model = YOLO(str(self.model_path))
            
            # Check device
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if device == 'cuda':
                print(f"   Device: {torch.cuda.get_device_name(0)}")
            else:
                print(f"   Device: CPU")
            
            print(f"✅ Model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to load model: {str(e)}")
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            return False
    
    def get_image_files(self):
        """Get all image files from folder"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
        image_files = [
            f for f in self.image_folder.iterdir()
            if f.suffix.lower() in image_extensions
        ]
        return sorted(image_files)
    
    def run_auto_labeling(self, conf_threshold=0.25, iou_threshold=0.45):
        """Run auto-labeling on all images"""
        try:
            print(f"\n{'='*80}")
            print(f"🚀 Starting Auto-Labeling")
            print(f"{'='*80}\n")
            
            # Get image files
            image_files = self.get_image_files()
            
            if not image_files:
                print("❌ No image files found in selected folder")
                messagebox.showerror("Error", "No image files found in the selected folder")
                return False
            
            print(f"📊 Found {len(image_files)} images to label\n")
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            start_time = time.time()
            successful = 0
            failed = 0
            total_detections = 0
            
            # Process each image
            for image_file in tqdm(image_files, desc="Processing images"):
                try:
                    # Run inference
                    results = self.model.predict(
                        source=str(image_file),
                        conf=conf_threshold,
                        iou=iou_threshold,
                        device=device,
                        verbose=False
                    )
                    
                    if results and len(results) > 0:
                        result = results[0]
                        
                        # Get image dimensions
                        img_height = result.orig_img.shape[0]
                        img_width = result.orig_img.shape[1]
                        
                        # Extract boxes
                        boxes = result.boxes
                        detections = len(boxes)
                        total_detections += detections
                        
                        # Create label file
                        label_file = self.output_folder / f"{image_file.stem}.txt"
                        
                        with open(label_file, 'w') as f:
                            for box in boxes:
                                # Get class and confidence
                                cls = int(box.cls[0])
                                conf = float(box.conf[0])
                                
                                # Get normalized coordinates (YOLO format)
                                x_center = float(box.xywh[0][0]) / img_width
                                y_center = float(box.xywh[0][1]) / img_height
                                width = float(box.xywh[0][2]) / img_width
                                height = float(box.xywh[0][3]) / img_height
                                
                                # Clamp values to [0, 1]
                                x_center = max(0, min(1, x_center))
                                y_center = max(0, min(1, y_center))
                                width = max(0, min(1, width))
                                height = max(0, min(1, height))
                                
                                # Write to file in YOLO format: class x_center y_center width height
                                f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                        
                        successful += 1
                
                except Exception as e:
                    print(f"\n⚠️  Failed to process {image_file.name}: {str(e)}")
                    failed += 1
            
            elapsed_time = time.time() - start_time
            
            print(f"\n{'='*80}")
            print(f"✅ Auto-Labeling Complete!")
            print(f"{'='*80}\n")
            print(f"📊 Statistics:")
            print(f"   Total images: {len(image_files)}")
            print(f"   Successfully labeled: {successful}")
            print(f"   Failed: {failed}")
            print(f"   Total detections: {total_detections}")
            print(f"   Processing time: {elapsed_time:.2f}s")
            print(f"   Average time per image: {elapsed_time/len(image_files):.2f}s")
            print(f"\n💾 Labels saved to: {self.output_folder}\n")
            
            messagebox.showinfo(
                "Success",
                f"Auto-labeling complete!\n\n"
                f"Successfully labeled: {successful}/{len(image_files)}\n"
                f"Total detections: {total_detections}\n"
                f"Processing time: {elapsed_time:.2f}s\n\n"
                f"Labels saved to:\n{self.output_folder}"
            )
            
            return True
            
        except Exception as e:
            print(f"❌ Auto-labeling failed: {str(e)}")
            messagebox.showerror("Error", f"Auto-labeling failed:\n{str(e)}")
            return False
    
    def show_config_dialog(self):
        """Show dialog for confidence and IOU thresholds"""
        config_window = tk.Tk()
        config_window.title("Auto-Labeling Configuration")
        config_window.geometry("300x200")
        
        # Confidence threshold
        tk.Label(config_window, text="Confidence Threshold:").pack(pady=5)
        conf_var = tk.DoubleVar(value=0.25)
        tk.Scale(config_window, from_=0.0, to=1.0, resolution=0.05, 
                orient=tk.HORIZONTAL, variable=conf_var).pack(fill=tk.X, padx=10)
        
        # IOU threshold
        tk.Label(config_window, text="IOU Threshold:").pack(pady=5)
        iou_var = tk.DoubleVar(value=0.45)
        tk.Scale(config_window, from_=0.0, to=1.0, resolution=0.05, 
                orient=tk.HORIZONTAL, variable=iou_var).pack(fill=tk.X, padx=10)
        
        result = {'ok': False, 'conf': 0.25, 'iou': 0.45}
        
        def on_ok():
            result['conf'] = conf_var.get()
            result['iou'] = iou_var.get()
            result['ok'] = True
            config_window.quit()
        
        def on_cancel():
            config_window.quit()
        
        button_frame = tk.Frame(config_window)
        button_frame.pack(pady=20)
        
        tk.Button(button_frame, text="Start", command=on_ok, width=10).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Cancel", command=on_cancel, width=10).pack(side=tk.LEFT, padx=5)
        
        config_window.mainloop()
        
        return result
    
    def run(self):
        """Main execution flow"""
        print(f"\n{'='*80}")
        print(f"🏷️  YOLO Auto-Labeling Tool")
        print(f"{'='*80}\n")
        
        # Step 1: Select model
        if not self.select_model():
            return
        
        # Step 2: Select image folder
        if not self.select_image_folder():
            return
        
        # Step 3: Select output folder
        if not self.select_output_folder():
            return
        
        # Step 4: Load model
        if not self.load_model():
            return
        
        # Step 5: Show configuration dialog
        config = self.show_config_dialog()
        if not config['ok']:
            print("❌ Auto-labeling cancelled")
            return
        
        # Step 6: Run auto-labeling
        self.run_auto_labeling(conf_threshold=config['conf'], iou_threshold=config['iou'])


def main():
    """Main entry point"""
    try:
        auto_label = AutoLabelGUI()
        auto_label.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Auto-labeling interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        messagebox.showerror("Error", f"An error occurred:\n{str(e)}")


if __name__ == "__main__":
    main()

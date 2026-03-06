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
        """Load YOLO model and display its classes"""
        try:
            print(f"\n📦 Loading model: {self.model_path.name}")
            self.model = YOLO(str(self.model_path))
            
            # Check device
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if device == 'cuda':
                print(f"   Device: {torch.cuda.get_device_name(0)}")
            else:
                print(f"   Device: CPU")
            
            # Display model classes
            class_names = self.model.names  # dict: {0: 'fire', 1: 'smoke', ...}
            print(f"   Classes in model ({len(class_names)}):")
            for cls_id, cls_name in class_names.items():
                print(f"     {cls_id}: {cls_name}")
            
            self.model_class_names = class_names
            self.class_mapping = None  # Will be set by user if needed
            
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
    
    def run_auto_labeling(self, conf_threshold=0.25, iou_threshold=0.45, class_mapping=None):
        """Run auto-labeling on all images
        
        Args:
            conf_threshold: Confidence threshold for detection
            iou_threshold: IOU threshold for NMS
            class_mapping: dict mapping model class IDs to output class IDs.
                          If None, all classes are exported with original IDs.
        """
        try:
            print(f"\n{'='*80}")
            print(f"🚀 Starting Auto-Labeling")
            print(f"{'='*80}\n")
            
            if class_mapping:
                print(f"📋 Class mapping (model ID → output ID):")
                for model_id, output_id in sorted(class_mapping.items()):
                    name = self.model_class_names.get(model_id, f"class_{model_id}")
                    print(f"   {model_id} ({name}) → {output_id}")
                print()
            
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
                        
                        # Create label file
                        label_file = self.output_folder / f"{image_file.stem}.txt"
                        
                        with open(label_file, 'w') as f:
                            for box in boxes:
                                # Get class and confidence
                                cls = int(box.cls[0])
                                conf = float(box.conf[0])
                                
                                # Apply class mapping: skip classes not in mapping
                                if class_mapping is not None:
                                    if cls not in class_mapping:
                                        continue  # Skip this class
                                    output_cls = class_mapping[cls]
                                else:
                                    output_cls = cls
                                
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
                                f.write(f"{output_cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                                total_detections += 1
                        
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
        """Show dialog for confidence, IOU thresholds, and class selection"""
        config_window = tk.Tk()
        config_window.title("Auto-Labeling Configuration")
        config_window.geometry("500x550")
        
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
        
        # Class selection
        tk.Label(config_window, text="─" * 60).pack()
        tk.Label(config_window, text="Select classes to export (and remap output IDs):",
                font=("Arial", 10, "bold")).pack(pady=5)
        tk.Label(config_window, text="Check classes to include. 'Output ID' is the class ID written to label files.",
                font=("Arial", 8)).pack()
        
        # Scrollable frame for classes
        class_frame = tk.Frame(config_window)
        class_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        canvas = tk.Canvas(class_frame, height=200)
        scrollbar = tk.Scrollbar(class_frame, orient="vertical", command=canvas.yview)
        scrollable = tk.Frame(canvas)
        
        scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Header
        header = tk.Frame(scrollable)
        header.pack(fill=tk.X, pady=2)
        tk.Label(header, text="Include", width=8, font=("Arial", 9, "bold")).pack(side=tk.LEFT)
        tk.Label(header, text="Model Class", width=25, anchor="w", font=("Arial", 9, "bold")).pack(side=tk.LEFT)
        tk.Label(header, text="Output ID", width=10, font=("Arial", 9, "bold")).pack(side=tk.LEFT)
        
        class_vars = {}  # {model_cls_id: (check_var, output_id_var)}
        for cls_id, cls_name in sorted(self.model_class_names.items()):
            row = tk.Frame(scrollable)
            row.pack(fill=tk.X, pady=1)
            
            check_var = tk.BooleanVar(value=True)  # All selected by default
            tk.Checkbutton(row, variable=check_var, width=5).pack(side=tk.LEFT)
            tk.Label(row, text=f"{cls_id}: {cls_name}", width=25, anchor="w").pack(side=tk.LEFT)
            
            output_id_var = tk.StringVar(value=str(cls_id))  # Default: same as model ID
            tk.Entry(row, textvariable=output_id_var, width=8).pack(side=tk.LEFT, padx=5)
            
            class_vars[cls_id] = (check_var, output_id_var)
        
        result = {'ok': False, 'conf': 0.25, 'iou': 0.45, 'class_mapping': None}
        
        def on_ok():
            result['conf'] = conf_var.get()
            result['iou'] = iou_var.get()
            
            # Build class mapping: {model_cls_id: output_cls_id} for selected classes
            mapping = {}
            for cls_id, (check_var, output_id_var) in class_vars.items():
                if check_var.get():
                    try:
                        output_id = int(output_id_var.get())
                        mapping[cls_id] = output_id
                    except ValueError:
                        messagebox.showerror("Error", f"Invalid output ID for class {cls_id}: {output_id_var.get()}")
                        return
            
            if not mapping:
                messagebox.showerror("Error", "At least one class must be selected!")
                return
            
            result['class_mapping'] = mapping
            result['ok'] = True
            config_window.quit()
        
        def on_cancel():
            config_window.quit()
        
        button_frame = tk.Frame(config_window)
        button_frame.pack(pady=10)
        
        tk.Button(button_frame, text="Start", command=on_ok, width=10).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Cancel", command=on_cancel, width=10).pack(side=tk.LEFT, padx=5)
        
        config_window.mainloop()
        config_window.destroy()
        
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
        self.run_auto_labeling(
            conf_threshold=config['conf'], 
            iou_threshold=config['iou'],
            class_mapping=config.get('class_mapping')
        )


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

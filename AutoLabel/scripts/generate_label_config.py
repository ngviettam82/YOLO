#!/usr/bin/env python3
"""
Generate Label Studio Label Configuration Template
Creates the XML configuration that users need to paste into Label Studio
"""

import sys
from pathlib import Path
import yaml
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))


def load_class_names():
    """Load class names from dataset config"""
    data_yaml = Path(PROJECT_ROOT) / "dataset" / "data.yaml"
    class_names = {}
    
    if data_yaml.exists():
        try:
            with open(data_yaml, 'r') as f:
                data = yaml.safe_load(f)
                if data and 'names' in data:
                    class_names = data['names']
        except Exception as e:
            print(f"Could not load class names: {e}")
    
    return class_names


def generate_xml_config(class_names):
    """Generate Label Studio XML configuration"""
    colors = ["green", "blue", "red", "yellow", "purple", "orange", "cyan", "magenta", "lime", "pink"]
    
    # Build Label elements
    labels_xml = ""
    for class_id, class_name in sorted(class_names.items()):
        color = colors[class_id % len(colors)]
        labels_xml += f'    <Label value="{class_name}" background="{color}"/>\n'
    
    config = f"""<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
{labels_xml}  </RectangleLabels>
</View>"""
    return config


def show_gui():
    """Show GUI with XML template and instructions"""
    class_names = load_class_names()
    
    if not class_names:
        messagebox.showerror("Error", "Could not load class names from data.yaml")
        return
    
    xml_config = generate_xml_config(class_names)
    
    # Create window
    root = tk.Tk()
    root.title("Label Studio Label Configuration Generator")
    root.geometry("800x600")
    
    # Header
    header = tk.Label(root, text="📋 Label Studio Configuration Template", font=("Arial", 14, "bold"), fg="blue")
    header.pack(pady=10)
    
    # Instructions
    instructions = tk.Label(root, text="""
Follow these steps:
1. Copy the XML configuration below
2. Go to Label Studio → Project Settings → Labeling Interface
3. Paste the XML code into the text editor
4. Click "Save" to apply the configuration

Your class names from data.yaml:
""", justify=tk.LEFT, wraplength=700)
    instructions.pack(padx=10, pady=5)
    
    # Show class names
    class_list = tk.Label(root, text="\n".join([f"  • {name}" for name in class_names.values()]), 
                         fg="darkgreen", font=("Arial", 10))
    class_list.pack(pady=5)
    
    # XML Config
    config_label = tk.Label(root, text="XML Configuration (Copy this):", font=("Arial", 11, "bold"))
    config_label.pack(padx=10, pady=(10, 5))
    
    # Text area with config
    text_frame = tk.Frame(root)
    text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
    scrollbar = tk.Scrollbar(text_frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    text_widget = scrolledtext.ScrolledText(text_frame, height=12, width=80, 
                                           yscrollcommand=scrollbar.set, font=("Courier", 9))
    text_widget.pack(fill=tk.BOTH, expand=True)
    scrollbar.config(command=text_widget.yview)
    
    text_widget.insert(tk.END, xml_config)
    text_widget.config(state=tk.DISABLED)  # Read-only
    
    # Copy button
    def copy_to_clipboard():
        root.clipboard_clear()
        root.clipboard_append(xml_config)
        messagebox.showinfo("Success", "Configuration copied to clipboard!")
    
    copy_btn = tk.Button(root, text="📋 Copy to Clipboard", command=copy_to_clipboard, 
                        bg="lightblue", font=("Arial", 11))
    copy_btn.pack(pady=10)
    
    # Exit button
    exit_btn = tk.Button(root, text="Close", command=root.quit, font=("Arial", 11))
    exit_btn.pack(pady=5)
    
    root.mainloop()


def main():
    """Main entry point"""
    print("\n" + "="*80)
    print("📋 Label Studio Configuration Generator")
    print("="*80 + "\n")
    
    class_names = load_class_names()
    
    if not class_names:
        print("❌ Could not load class names from dataset/data.yaml")
        return
    
    print(f"✓ Loaded {len(class_names)} class names:")
    for class_id, name in sorted(class_names.items()):
        print(f"  • {name}")
    
    print("\n📋 Generated XML Configuration:\n")
    xml_config = generate_xml_config(class_names)
    print(xml_config)
    
    print("\n" + "="*80)
    print("🚀 How to use:")
    print("="*80)
    print("1. Copy the XML configuration above")
    print("2. Go to Label Studio → Project Settings → Labeling Interface")
    print("3. Paste the XML code into the text editor")
    print("4. Click 'Save' to apply the configuration")
    print("5. Now your imported labels will be recognized!\n")
    
    # Show GUI option
    show_gui_response = input("Would you like to see the GUI version? (y/n): ").strip().lower()
    if show_gui_response == 'y':
        show_gui()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        messagebox.showerror("Error", f"An error occurred:\n{str(e)}")

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import glob
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2_video_predictor
import shutil
import time  

# Set random seed
np.random.seed(3)

# Environment variable setting
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Get the absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Helper functions
def extract_numbers(file_name):
    numbers = re.findall(r'\d+', file_name)
    return tuple(map(int, numbers))

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# User-customizable parameters
class Config:
    def __init__(self):
        self.input_dir = ""
        self.output_dir = ""
        self.checkpoint = ""
        self.model_cfg = ""
        self.input_points = []
        self.input_labels = []

class SAM2App:
    def __init__(self, master):
        self.master = master
        self.master.title("SAM2 Mask Generator")
        self.config = Config()
        matplotlib.use('TkAgg')
        self.current_image = None
        self.current_mask = None
        self.plotting_mode = False
        self.processing = False
        self.device = 'cpu'
        self.point_mode = 'foreground'
        self.inference_state = None
        
        self.setup_gui()
        self.prev_masks = []
        self.current_masks = [None, None, None]
        self.processTimes = 0
        self.start_index = 1
        self.end_index = []
        self.start_index_previous = []
        self.end_index_previous = []
    
    def setup_gui(self):
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Top frame for input/output selection
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))

        self.setup_io_frame(top_frame)

        # Add range selection controls - moved after input/output selection
        range_frame = ttk.Frame(main_frame)
        range_frame.pack(fill=tk.X, pady=(0, 10))  # Modified pady to ensure proper spacing
        ttk.Label(range_frame, text="Start frame#:").pack(side=tk.LEFT, padx=(0, 5))
        self.start_index_entry = ttk.Entry(range_frame, width=5)
        self.start_index_entry.pack(side=tk.LEFT, padx=(0, 5))
        # Add binding events
        self.start_index_entry.bind('<Return>', lambda e: self.show_raw_image())
        self.start_index_entry.bind('<FocusOut>', lambda e: self.show_raw_image())
        
        ttk.Label(range_frame, text="End frame#:").pack(side=tk.LEFT, padx=(0, 5))
        self.end_index_entry = ttk.Entry(range_frame, width=5)
        self.end_index_entry.pack(side=tk.LEFT, padx=(0, 5))

        # Middle frame for controls
        middle_frame = ttk.Frame(main_frame)
        middle_frame.pack(fill=tk.X, pady=(0, 10))

        # Visualization controls
        viz_frame = ttk.LabelFrame(middle_frame, text="Visualization Controls")
        viz_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        self.setup_viz_controls(viz_frame)

        # Analysis controls
        analysis_frame = ttk.LabelFrame(middle_frame, text="Analysis Controls")
        analysis_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

        self.setup_analysis_controls(analysis_frame)

        # Progress bar and label
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        self.setup_progress_bar(progress_frame)

        # Image frame: original image and mask display
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(fill=tk.BOTH, expand=True)
        self.setup_image_display(image_frame)

    def setup_io_frame(self, parent):
        input_frame = ttk.Frame(parent)
        input_frame.pack(fill=tk.X, pady=5)
        ttk.Label(input_frame, text="Input Directory:").pack(side=tk.LEFT, padx=(0, 5))
        self.input_entry = ttk.Entry(input_frame)
        self.input_entry.pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(input_frame, text="Browse", command=self.select_input).pack(side=tk.LEFT, padx=(5, 0))

        output_frame = ttk.Frame(parent)
        output_frame.pack(fill=tk.X, pady=5)
        ttk.Label(output_frame, text="Output Directory:").pack(side=tk.LEFT, padx=(0, 5))
        self.output_entry = ttk.Entry(output_frame)
        self.output_entry.pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(output_frame, text="Browse", command=self.select_output).pack(side=tk.LEFT, padx=(5, 0))

    def setup_viz_controls(self, parent):
        self.plot_points_btn = ttk.Button(parent, text="Start Drawing Points",
                                          command=self.toggle_plotting_mode)
        self.plot_points_btn.pack(side=tk.LEFT, padx=5, pady=5)

        self.toggle_mode_btn = ttk.Button(parent, text="Mode: Foreground",
                                          command=self.toggle_point_mode)
        self.toggle_mode_btn.pack(side=tk.LEFT, padx=5, pady=5)

        self.clear_points_btn = ttk.Button(parent, text="Clear Points", command=self.clear_points)
        self.clear_points_btn.pack(side=tk.LEFT, padx=5, pady=5)

    def setup_analysis_controls(self, parent):
        device_frame = ttk.Frame(parent)
        device_frame.pack(fill=tk.X, pady=5)
        ttk.Label(device_frame, text="Device:").pack(side=tk.LEFT, padx=(0, 5))
        self.device_var = tk.StringVar(self.master)
        self.device_var.set("CPU")
        self.device_menu = ttk.Combobox(device_frame, textvariable=self.device_var,
                                        values=["CPU", "GPU"])
        self.device_menu.pack(side=tk.LEFT, padx=(0, 10))
        self.device_menu.bind("<<ComboboxSelected>>", self.on_device_change)

        self.model_var = tk.StringVar(self.master)
        self.model_var.set("SAM2 Hiera Large")
        ttk.Label(device_frame, text="Model:").pack(side=tk.LEFT, padx=(0, 5))
        self.model_menu = ttk.Combobox(device_frame, textvariable=self.model_var,
                                       values=["SAM2 Hiera Large", "SAM2 Hiera Base Plus",
                                               "SAM2 Hiera Small", "SAM2 Hiera Tiny"])
        self.model_menu.pack(side=tk.LEFT, padx=(0, 10))

        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=5)
        self.start_btn = ttk.Button(control_frame, text="Start Processing", command=self.start_processing)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 5))
        self.stop_btn = ttk.Button(control_frame, text="Stop Processing", command=self.stop_processing,
                                   state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 5))

    def setup_progress_bar(self, parent):
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(parent, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X)
        self.progress_label = ttk.Label(parent, text="")
        self.progress_label.pack()

    def setup_image_display(self, parent):
        self.fig, axes_2d  = plt.subplots(2, 2, figsize=(8, 6))
        (self.ax1, self.ax2, self.ax3, self.ax4) = axes_2d.ravel()

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)


    def select_input(self):
        self.config.input_dir = filedialog.askdirectory(title="Select input folder")
        if self.config.input_dir:
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, self.config.input_dir)
            self.cleanup_temp_folders()  # Clean up temporary folders after confirming input folder
            self.show_raw_image()

    def select_output(self):
        self.config.output_dir = filedialog.askdirectory(title="Select output folder")
        if self.config.output_dir:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, self.config.output_dir)

    def on_device_change(self, event):
        selected_device = self.device_var.get()
        if selected_device == "GPU":
            if torch.cuda.is_available():
                self.device = 'cuda'
                print("GPU selected and available. Using CUDA.")
            else:
                self.device = 'cpu'
                self.device_var.set("CPU")
                messagebox.showwarning("GPU Not Available",
                                       "GPU is not available. Switching to CPU.")
                print("GPU not available. Falling back to CPU.")
        else:
            self.device = 'cpu'
            print("CPU selected.")

    def show_raw_image(self):
        image_files = self.get_image_files(self.config.input_dir)
        if image_files:
            self.start_index = int(self.start_index_entry.get() or 1)
            try:
                # Try to read image with OpenCV
                self.current_image = cv2.imread(image_files[0 + self.start_index - 1])
                
                # If reading fails (e.g., 32-bit TIFF), try using PIL/Pillow
                if self.current_image is None:

                    
                    # Use PIL to read image
                    pil_image = Image.open(image_files[0 + self.start_index - 1])
                    
                    # Convert to RGB mode (if not already)
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    
                    # Convert to numpy array
                    self.current_image = np.array(pil_image)
                    
                    # PIL reads images in RGB order, no need for BGR to RGB conversion
                else:
                    # OpenCV reads images in BGR order, need to convert to RGB
                    self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
                
                self.display_image()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
                print(f"Error loading image: {e}")

    def display_image(self, keep_points=False):
        if self.current_image is not None:
            self.ax1.clear()
            self.ax1.imshow(self.current_image)
            self.ax1.set_title('Input Image')
            self.ax1.set_xlabel('X')
            self.ax1.set_ylabel('Y')

            # Always check if points should be retained
            if self.config.input_points:
                self.update_image_with_points(self.ax1)

            if all(item is None for item in self.current_masks):
                self.ax2.clear()
                self.ax2.set_title('Mask will be displayed here')
                self.ax2.axis('off')
            else:

                # ax2  mask1
                self.ax2.clear()
                self.ax2.imshow(self.current_masks[0], cmap='gray')
                self.ax2.set_title('Mask #1')
                self.ax2.axis('off')

                # ax3 mask2
                # self.ax3.clear()
                # self.ax3.imshow(self.current_masks[1], cmap='gray')
                # self.ax3.set_title('Mask #2')
                # self.ax3.axis('off')

                # # ax4 mask3
                # self.ax4.clear()
                # self.ax4.imshow(self.current_masks[2], cmap='gray')
                # self.ax4.set_title('Mask #3')
                # self.ax4.axis('off')



            self.canvas.draw()

    def toggle_plotting_mode(self):
        self.plotting_mode = not self.plotting_mode
        if self.plotting_mode:
            self.plot_points_btn.config(text="Stop Drawing")
            self.canvas.mpl_connect('button_press_event', self.on_click)
        else:
            self.plot_points_btn.config(text="Start Drawing Points")

    def toggle_point_mode(self):
        self.point_mode = 'background' if self.point_mode == 'foreground' else 'foreground'
        self.toggle_mode_btn.config(text=f"Mode: {self.point_mode.capitalize()}")

    def on_click(self, event):
        if event.inaxes == self.ax1 and self.plotting_mode:
            x, y = int(event.xdata), int(event.ydata)
            if len(self.config.input_points) < 100:
                self.config.input_points.append([x, y])
                self.config.input_labels.append(1 if self.point_mode == 'foreground' else 0)
                # Update image and retain points
                self.display_image(keep_points=True)
            else:
                messagebox.showwarning("Warning", "Maximum number of points (100) reached.")

    def clear_points(self):
        self.config.input_points = []
        self.config.input_labels = []
        self.display_image(keep_points=True)
        messagebox.showinfo("Info", "All points have been cleared.")

    def update_image_with_points(self, ax=None):
        if ax is None:
            ax = self.ax1

        ax.clear()  # Clear the axes before redrawing
        ax.imshow(self.current_image)  # Redraw the current image

        if self.config.input_points:
            points = np.array(self.config.input_points)
            labels = np.array(self.config.input_labels)
            ax.scatter(points[labels == 1, 0], points[labels == 1, 1], color='red', s=50, label='Foreground')
            ax.scatter(points[labels == 0, 0], points[labels == 0, 1], color='blue', s=50, label='Background')

        # Clear the previous legend to avoid repetition
        legend = ax.get_legend()
        if legend:
            legend.remove()

        # Draw the new legend
        ax.legend(loc='upper right', fontsize='small')

        self.canvas.draw()

    def update_progress(self, progress):
        self.progress_var.set(progress)
        self.progress_label.config(text=f"{progress:.1f}%")
        self.master.update()

    def start_processing(self):
        if not self.config.input_dir or not self.config.output_dir:
            messagebox.showerror("Error", "Please select both input and output folders.")
            return

        if not self.config.input_points:
            messagebox.showerror("Error", "Please add at least one point.")
            return

        # Set up model
        models = {
            "SAM2 Hiera Large": ("checkpoints/sam2.1_hiera_large.pt", "sam2/configs/sam2.1/sam2.1_hiera_l.yaml"),
            "SAM2 Hiera Base Plus": ("checkpoints/sam2.1_hiera_base_plus.pt", "sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"),
            "SAM2 Hiera Small": ("checkpoints/sam2.1_hiera_small.pt", "sam2/configs/sam2.1/sam2.1_hiera_s.yaml"),
            "SAM2 Hiera Tiny": ("checkpoints/sam2.1_hiera_tiny.pt", "sam2/configs/sam2.1/sam2.1_hiera_t.yaml")
        }
        self.config.checkpoint, self.config.model_cfg = models[self.model_var.get()]

        # Start processing
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.processing = True
        self.process_images()

    def stop_processing(self):
        self.processing = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.progress_label.config(text="Processing stopped.")

    def process_images(self):
        # Ensure output directory exists
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Get parent directory and folder name of input_dir
        parent_dir = os.path.dirname(self.config.input_dir)
        input_folder_name = os.path.basename(self.config.input_dir)
        
        # Get image file list and sort
        image_files = self.get_image_files(self.config.input_dir)
        total_images = len(image_files)
        
        # Get user input range
        self.start_index = int(self.start_index_entry.get() or 1)
        self.end_index = int(self.end_index_entry.get() or total_images)
        
        # Ensure indices are within valid range
        self.start_index = max(1, self.start_index)
        self.end_index = min(total_images+1, self.end_index)
                
        # Create temporary directory, use input folder name with suffix, if range hasn't changed twice, use the previous jpg_range folder
        if self.start_index != self.start_index_previous or self.end_index != self.end_index_previous:
            self.processTimes = self.processTimes + 1
        temp_dir = os.path.join(parent_dir, f"{input_folder_name}_jpg/{input_folder_name}_jpg_range{self.processTimes}")
        os.makedirs(temp_dir, exist_ok=True)

        self.start_index_previous = self.start_index
        self.end_index_previous = self.end_index

        # Check JPEG files in temporary folder
        existing_jpegs = {os.path.basename(f): f for f in glob.glob(os.path.join(temp_dir, "*.jpg"))}
        
        # Convert all images to JPEG format
        jpeg_files = []
        for idx, img_path in enumerate(image_files[self.start_index-1:self.end_index]):
            # Use simple numeric sequence naming for temporary files
            jpeg_name = f"{idx+self.start_index:06d}.jpg"  # Generate filename like "000001.jpg"
            jpeg_path = os.path.join(temp_dir, jpeg_name)
        
            # Check if corresponding JPEG file already exists
            if jpeg_name in existing_jpegs:
                jpeg_files.append(jpeg_path)
                # Update progress
                progress = idx / total_images * 50
                self.update_progress(progress)
                self.progress_label.config(text=f"Using existing JPEG: {idx+1}/{total_images}")
                continue
            
            # Update progress
            progress = idx / total_images * 50
            self.update_progress(progress)
            self.progress_label.config(text=f"Converting to JPEG: {idx+1}/{total_images}")
            
            try:
                # First try to read with OpenCV
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                
                # If OpenCV reading fails, try with PIL/Pillow
                if img is None:
                    
                    print(f"OpenCV failed to read image: {img_path}. Trying with PIL...")
                    
                    # Use PIL to read image
                    pil_image = Image.open(img_path)
                    
                    # Extract bit depth information for logging
                    bit_depth = getattr(pil_image, 'bit_depth', 'unknown')
                    print(f"Image format: {pil_image.format}, Mode: {pil_image.mode}, Bit depth: {bit_depth}")
                    
                    # Special handling for 32-bit floating point TIFF and other high bit depth images
                    if pil_image.mode in ['F', 'I', 'I;16', 'I;32']:
                        # Convert to numpy array
                        img_array = np.array(pil_image)
                        
                        # Normalize to 0-255 range
                        min_val = np.min(img_array)
                        max_val = np.max(img_array)
                        
                        if max_val > min_val:  # Avoid division by zero
                            img = ((img_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                        else:
                            img = np.zeros_like(img_array, dtype=np.uint8)
                        
                        # If single channel image, convert to RGB
                        if len(img.shape) == 2:
                            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    else:
                        # For standard modes, convert to RGB
                        pil_image = pil_image.convert('RGB')
                        img = np.array(pil_image)
                        # PIL's RGB order is different from OpenCV's BGR order, need conversion
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                else:
                    # OpenCV successfully read, handle different data types
                    if img.dtype == np.uint16:
                        img = (img / 256).astype(np.uint8)
                    elif img.dtype == np.float32 or img.dtype == np.float64:
                        min_val = np.min(img)
                        max_val = np.max(img)
                        if max_val > min_val:
                            img = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                        else:
                            img = np.zeros_like(img, dtype=np.uint8)
                    
                    # Ensure image is colored
                    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
                # Save as JPEG
                cv2.imwrite(jpeg_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                jpeg_files.append(jpeg_path)
                
            except Exception as e:
                print(f"Failed to process image {img_path}: {str(e)}")
                # If processing fails, create a black image as placeholder
                try:
                    blank_img = np.zeros((100, 100, 3), dtype=np.uint8)  # Create a small black image
                    cv2.imwrite(jpeg_path, blank_img)
                    jpeg_files.append(jpeg_path)
                    print(f"Created blank placeholder for {img_path}")
                except:
                    print(f"Failed to create placeholder for {img_path}")
                continue

        sam2_checkpoint = os.path.join(current_dir, self.config.checkpoint)
        model_cfg = os.path.join(current_dir, self.config.model_cfg)
        
        # Initialize SAM2 model, use temporary JPEG directory
        predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
        
        # If previous state exists, reset first
        if self.inference_state is not None:
            predictor.reset_state(self.inference_state)
        
        # Initialize new state
        self.inference_state = predictor.init_state(video_path=temp_dir)

        # Convert input points and labels to numpy arrays
        points_array = np.array(self.config.input_points)
        labels_array = np.array(self.config.input_labels)
        
        # Call predictor to add new points or box
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=0,
            obj_id=1,
            points=points_array,  # Use converted points_array
            labels=labels_array,  # Use converted labels_array
        )

        # Add time recording variables
        inference_times = []  # Store inference time for each image
        total_start_time = time.time()  # Overall start time
        
        # run propagation throughout the video and collect the results in a dict
        video_segments = {}  # video_segments contains the per-frame segmentation results
        
        # Create inference time recording generator wrapper
        def timed_propagate_in_video():
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(self.inference_state):
                yield out_frame_idx, out_obj_ids, out_mask_logits
        
        # Start processing each frame
        frame_count = 0
        for out_frame_idx, out_obj_ids, out_mask_logits in timed_propagate_in_video():
            # Record single frame inference start time (here records the time point when propagate_in_video returns results)
            frame_inference_start = time.time()
            
            # Read image
            image = cv2.imread(image_files[out_frame_idx])
            if image is None:
                print(f"Failed to read image: {image_files[out_frame_idx]}. Skipping.")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_name = os.path.basename(image_files[out_frame_idx+self.start_index-1])

            # Process inference results (this part is the core post-inference processing)
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

            out_masks = video_segments[out_frame_idx]
            binary_masks = out_masks[1]
            binary_masks = binary_masks.astype(np.uint8) * 255
            
            # Record single frame inference end time
            frame_inference_end = time.time()
            frame_inference_time = frame_inference_end - frame_inference_start
            inference_times.append(frame_inference_time)
            frame_count += 1
            
            # Calculate current average inference time
            current_avg_time = sum(inference_times) / len(inference_times)
            current_fps = 1.0 / current_avg_time if current_avg_time > 0 else 0
            
            # Update progress with timing information
            progress = (out_frame_idx) / total_images * 100
            self.update_progress(progress)
            self.progress_label.config(
                text=f"Processing: {image_name} ({out_frame_idx+1}/{total_images}) - "
                     f"Frame Time: {frame_inference_time:.3f}s - Avg FPS: {current_fps:.1f}"
            )
            self.master.update()

            if not self.processing:
                print("Processing stopped by user.")
                break

            # Update current image display
            self.current_image = image
            self.display_image(keep_points=True)

            # Update current mask display
            self.current_masks = binary_masks
            self.display_image()

            # Save the binary mask
            mask1_dir = os.path.join(self.config.output_dir, "mask1")
            os.makedirs(mask1_dir, exist_ok=True)
            cv2.imwrite(os.path.join(mask1_dir, image_name), binary_masks[0])

            self.master.update()

        # Calculate and output final statistics
        total_end_time = time.time()
        total_processing_time = total_end_time - total_start_time
        
        if inference_times:
            avg_inference_time = sum(inference_times) / len(inference_times)
            avg_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
            min_time = min(inference_times)
            max_time = max(inference_times)
            
            # Print detailed time statistics
            print("\n" + "="*60)
            print("Inference Time Statistics Report")
            print("="*60)
            print(f"Number of processed images: {len(inference_times)}")
            print(f"Average inference time: {avg_inference_time:.4f} seconds")
            print(f"Average FPS: {avg_fps:.2f}")
            print(f"Fastest inference time: {min_time:.4f} seconds")
            print(f"Slowest inference time: {max_time:.4f} seconds")
            print(f"Total processing time: {total_processing_time:.2f} seconds")
            print("="*60)
            
            # Save time statistics to file
            stats_file = os.path.join(self.config.output_dir, "inference_timing_stats.txt")
            with open(stats_file, 'w', encoding='utf-8') as f:
                f.write("Inference Time Statistics Report\n")
                f.write("="*60 + "\n")
                f.write(f"Number of processed images: {len(inference_times)}\n")
                f.write(f"Average inference time: {avg_inference_time:.4f} seconds\n")
                f.write(f"Average FPS: {avg_fps:.2f}\n")
                f.write(f"Fastest inference time: {min_time:.4f} seconds\n")
                f.write(f"Slowest inference time: {max_time:.4f} seconds\n")
                f.write(f"Total processing time: {total_processing_time:.2f} seconds\n")
                f.write("="*60 + "\n")
                f.write("\nDetailed time records for each frame:\n")
                for i, t in enumerate(inference_times):
                    f.write(f"Frame {i+1}: {t:.4f}s\n")
            
            print(f"Detailed statistics saved to: {stats_file}")
        
        self.processing = False
        final_message = "Processing complete." if frame_count > 0 else "Processing stopped."
        if inference_times:
            final_message += f" Average inference time: {avg_inference_time:.3f}s"
        self.progress_label.config(text=final_message)

        # Re-enable start button and disable stop button when processing is done
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def get_image_files(self, directory):
        image_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.tif')
        image_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(image_extensions):
                    image_files.append(os.path.join(root, file))
        return sorted(image_files, key=lambda x: extract_numbers(os.path.basename(x)))

    def get_effective_points(self, image_shape):
        orig_points = np.array(self.config.input_points)
        orig_labels = np.array(self.config.input_labels)

        if len(orig_points) == 0 or len(self.prev_masks) == 0:
            return orig_points, orig_labels


        masks_to_check = self.prev_masks[-3:]

        effective_points = []
        effective_labels = []

        for p, lbl in zip(orig_points, orig_labels):
            x, y = p[0], p[1]

            if x < 0 or y < 0 or y >= image_shape[0] or x >= image_shape[1]:
                continue


            mask_values = [masks[y, x] for masks in masks_to_check]


            mask_values_bin = [1 if val > 127 else 0 for val in mask_values]

            num_fg = sum(mask_values_bin)
            num_bg = len(mask_values_bin) - num_fg


            user_bin = 1 if lbl == 1 else 0


            if num_fg == 3 and user_bin == 0:
                effective_points.append([x, y])
                effective_labels.append(1)
            elif num_bg == 3 and user_bin == 1:
                effective_points.append([x, y])
                effective_labels.append(0)
            else:

                if num_fg == 3 and user_bin == 1:
                    effective_points.append([x, y])
                    effective_labels.append(1)
                elif num_bg == 3 and user_bin == 0:
                    effective_points.append([x, y])
                    effective_labels.append(0)

        return np.array(effective_points), np.array(effective_labels)

    def cleanup_temp_folders(self):
        if not self.config.input_dir:
            return
        
        # Get parent directory and folder name of input_dir
        parent_dir = os.path.dirname(self.config.input_dir)
        input_folder_name = os.path.basename(self.config.input_dir)
        
        # jpg folder path
        jpg_dir = os.path.join(parent_dir, f"{input_folder_name}_jpg")
        
        # If jpg folder exists, delete all contents
        if os.path.exists(jpg_dir):
            try:
                for item in os.listdir(jpg_dir):
                    item_path = os.path.join(jpg_dir, item)
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                print(f"Cleaned up temporary folder: {jpg_dir}")
            except Exception as e:
                print(f"Error cleaning up temporary folder: {e}")
        else:
            # If jpg folder doesn't exist, create it
            os.makedirs(jpg_dir)
            print(f"Created temporary folder: {jpg_dir}")

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1000x800")
    style = ttk.Style()
    style.theme_use('clam')
    app = SAM2App(root)
    root.mainloop()
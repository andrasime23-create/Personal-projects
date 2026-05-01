import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
import os
import sys
import traceback

# Define the colors using Hex codes for Tkinter
BACKGROUND_COLOR = "#222E5B" # RGB(34, 46, 91)
BUTTON_COLOR = "#BDBDBD" # RGB(189, 189, 189)
TEXT_COLOR = "#FFFFFF" # White text for visibility
DISPLAY_FRAME_COLOR = "#1D274A" # Darker shade for the display area

# Global references for Tkinter widgets and image storage
root = None
instruction_label = None
image_display_label = None
intermediate_display_label = None 
results_text = None
mask_grid_frame = None 
import_button = None 
slider_value_label = None

# Global image storage (Numpy array)
CURRENT_IMAGE_PATH = None
CURRENT_IMAGE_BGR = None

# Global lists to prevent garbage collection of image references
original_tk_image = None
intermediate_tk_image = None
mask_tk_images = []

# Global variable for the adjustable mean intensity threshold (0-100)
THRESHOLD_MEAN = 40 

# --- Core Detection Functions ---

def det_day_night(img_bgr):
    """
    Classifies the image as 'Daytime' or 'Nighttime' based on mean intensity.
    """
    global THRESHOLD_MEAN
    
    if img_bgr is None:
        return "Undetermined", -1
    
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    global_mean_intensity = np.mean(img)
    
    # Use the threshold for automatic classification
    if global_mean_intensity > THRESHOLD_MEAN:
        classification = "Daytime (AUTO)"
    else:
        classification = "Nighttime (AUTO)"
    
    return classification, global_mean_intensity

def calculate_red_angle(img_rgb):
    """Calculates the Red Angle image plane for daytime segmentation."""
    img_rgb = img_rgb.astype(np.float32)
    R, G, B = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    denom = np.sqrt(R**2 + G**2 + B**2) + 1e-6 
    RedAngle = R / denom
    return RedAngle

def process_detection(img_bgr):
    """Runs classification and segmentation on the loaded BGR image data."""
    global THRESHOLD_MEAN
    
    if img_bgr is None:
        return None, None, [], "Error: No image data provided.", ""
    
    InputImage_RGB = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    classification, GlobalMean = det_day_night(img_bgr)
    
    log_messages = f"Mean Intensity Threshold: {THRESHOLD_MEAN}\n"
    log_messages += f"Image size: {InputImage_RGB.shape[1]}x{InputImage_RGB.shape[0]}\n"
    log_messages += f"Global Mean Intensity: {GlobalMean:.2f}\n"
    log_messages += f"Environment Detected: {classification}\n"
    
    IntermediateImage_RGB = None
    IntermediateTitle = ""
    OutBinaryImg = []
    
    if "Daytime" in classification:
        RedAngle = calculate_red_angle(InputImage_RGB)
        IntermediateImage = cv2.normalize(RedAngle, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        IntermediateTitle = "Red-Angle Image Plane"
        IntermediateImage_RGB = cv2.cvtColor(IntermediateImage, cv2.COLOR_GRAY2RGB)
        
        Mu, Sigma = np.mean(RedAngle), np.std(RedAngle)
        log_messages += f"Mean = {Mu:.4f}, Std = {Sigma:.4f}\n"
        
        SLIDER_CLIP_MAX = 0.54 + (THRESHOLD_MEAN / 100.0) * (0.90 - 0.54)
        log_messages += f"Segmentation Max Clip (based on slider): {SLIDER_CLIP_MAX:.4f}\n"

        for k in range(1, 8):
            M = Mu + k * Sigma
            M_clipped = np.clip(M, 0.54, SLIDER_CLIP_MAX) 
            BinaryImage = (RedAngle > M_clipped).astype(np.uint8) * 255
            OutBinaryImg.append((BinaryImage, f"Threshold k={k}"))
            log_messages += f"k={k}, Threshold M = {M_clipped:.4f}\n"

    elif "Nighttime" in classification:
        gray = cv2.cvtColor(InputImage_RGB, cv2.COLOR_RGB2GRAY)
        
        h, w = gray.shape
        base_size = min(h, w)
        kernel_size = max(3, int(base_size * 0.01))
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        log_messages += f"Adaptive kernel size: {kernel_size}x{kernel_size}\n"
        
        top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        bottom_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        IEG = cv2.add(gray, top_hat)
        IEG = cv2.subtract(IEG, bottom_hat)
        
        IEG = cv2.normalize(IEG, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        IntermediateTitle = f"Enhanced Grayscale (IEG) – Kernel {kernel_size}x{kernel_size}"
        IntermediateImage_RGB = cv2.cvtColor(IEG, cv2.COLOR_GRAY2RGB)
        
        thresholds = [70, 100, 130, 160, 190, 220, 250]
        for t in thresholds:
            _, BinaryImage = cv2.threshold(IEG, t, 255, cv2.THRESH_BINARY)
            OutBinaryImg.append((BinaryImage, f"Threshold t={t}"))
            log_messages += f"Threshold = {t}\n"
        
    else:
        return None, None, [], "Error: Unable to classify environment", ""
    
    return InputImage_RGB, IntermediateImage_RGB, OutBinaryImg, log_messages, IntermediateTitle

# --- Tkinter GUI Logic ---

def resize_and_convert(img_numpy, target_width, target_height):
    """Resizes a numpy image (RGB/BGR) and converts it to a Tkinter PhotoImage."""
    img_pil = Image.fromarray(img_numpy)
    
    ratio = min(target_width / img_pil.width, target_height / img_pil.height)
    new_width = int(img_pil.width * ratio)
    new_height = int(img_pil.height * ratio)
    
    resized_img = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(resized_img), new_width, new_height

def clear_mask_grid_frame():
    """Destroys all previous widgets in the mask display area."""
    for widget in mask_grid_frame.winfo_children():
        widget.destroy()
    global mask_tk_images
    mask_tk_images = []

def update_gui_with_results(Original_RGB, Intermediate_RGB, OutBinaryImg, log_messages, IntermediateTitle):
    """Handles updating all image and log widgets in the GUI."""
    global original_tk_image, intermediate_tk_image
    
    clear_mask_grid_frame()
    instruction_label.grid_forget() 
    
    # Adjusted sizes for a compact, non-scrolling layout
    TARGET_W, TARGET_H = 300, 220 
    MASK_W, MASK_H = 140, 110 # Smaller masks to fit 3x3 vertically
    
    # 1. Update Original Image
    original_tk_image, w1, h1 = resize_and_convert(Original_RGB, TARGET_W, TARGET_H)
    image_display_label.config(
        image=original_tk_image, text="Original Image", compound="top",
        fg=TEXT_COLOR, width=w1, height=h1 + 20 
    )
    image_display_label.image = original_tk_image
    
    # 2. Update Intermediate Image
    intermediate_tk_image, w2, h2 = resize_and_convert(Intermediate_RGB, TARGET_W, TARGET_H)
    intermediate_display_label.config(
        image=intermediate_tk_image, text=IntermediateTitle, compound="top",
        fg=TEXT_COLOR, width=w2, height=h2 + 20
    )
    intermediate_display_label.image = intermediate_tk_image
    
    # 3. Update Results Log (Text Box)
    results_text.config(state=tk.NORMAL)
    results_text.delete(1.0, tk.END)
    results_text.insert(tk.END, log_messages)
    results_text.config(state=tk.DISABLED)

    # 4. Display Binary Masks in 3x3 Grid
    mask_title_label = tk.Label(mask_grid_frame, text="SEGMENTATION MASKS", 
                                 font=("Arial", 14, "bold"), bg=BACKGROUND_COLOR, fg=TEXT_COLOR)
    mask_title_label.grid(row=0, column=0, columnspan=3, pady=5)
    
    for i, (mask_img, title) in enumerate(OutBinaryImg):
        row = (i // 3) + 1
        col = i % 3
        
        mask_rgb = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2RGB)
        tk_mask, w_m, h_m = resize_and_convert(mask_rgb, MASK_W, MASK_H)
        mask_tk_images.append(tk_mask)
        
        mask_label = tk.Label(
            mask_grid_frame, image=tk_mask, text=title, compound="top",
            bg=DISPLAY_FRAME_COLOR, fg=TEXT_COLOR, font=("Arial", 9)
        )
        mask_label.grid(row=row, column=col, padx=5, pady=5, sticky="n") # Reduced padding
        
    # Add placeholders if fewer than 9 masks were generated
    if len(OutBinaryImg) < 9:
         for i in range(len(OutBinaryImg), 9):
             row = (i // 3) + 1
             col = i % 3
             placeholder = tk.Label(mask_grid_frame, text="", bg=BACKGROUND_COLOR, width=20, height=10)
             placeholder.grid(row=row, column=col, padx=5, pady=5)

# --- Function to reprocess image when settings change ---
def reprocess_current_image(event=None):
    """
    Reruns the detection algorithm on the stored image data 
    and updates the GUI based on the current slider settings.
    """
    global CURRENT_IMAGE_BGR

    if CURRENT_IMAGE_BGR is not None:
        try:
            import_button.config(state=tk.DISABLED)

            results = process_detection(CURRENT_IMAGE_BGR)
            Original_RGB, Intermediate_RGB, OutBinaryImg, log_messages, IntermediateTitle = results
            
            if Original_RGB is not None:
                update_gui_with_results(Original_RGB, Intermediate_RGB, OutBinaryImg, log_messages, IntermediateTitle)
            else:
                print(log_messages)
                tk.messagebox.showerror("Error", log_messages)

        except Exception as e:
            error_message = f"An error occurred during real-time processing:\n{e}\n{traceback.format_exc()}"
            print(error_message)
            tk.messagebox.showerror("Processing Error", "An error occurred during real-time processing. See console for details.")
        
        finally:
            import_button.config(state=tk.NORMAL)

def update_threshold(new_value):
    """Updates the global threshold and the display label, then reprocesses the image."""
    global THRESHOLD_MEAN
    THRESHOLD_MEAN = int(float(new_value))
    slider_value_label.config(text=f"Current Threshold: {THRESHOLD_MEAN}")
    reprocess_current_image() # Trigger real-time update

def import_image():
    """Handles button click, loads image data, and runs initial detection."""
    global CURRENT_IMAGE_PATH, CURRENT_IMAGE_BGR
    
    file_path = filedialog.askopenfilename(
        title="Select an Image File",
        filetypes=(("Image files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp"), ("All files", "*.*"))
    )
    
    if file_path:
        CURRENT_IMAGE_PATH = file_path
        CURRENT_IMAGE_BGR = cv2.imread(file_path)

        if CURRENT_IMAGE_BGR is not None:
            reprocess_current_image()
        else:
            tk.messagebox.showerror("Error", "Could not read image file.")
            CURRENT_IMAGE_PATH = None
            CURRENT_IMAGE_BGR = None
        

def setup_gui():
    """Sets up the main window and all initial widgets."""
    global root, instruction_label, image_display_label, intermediate_display_label, results_text, mask_grid_frame, import_button, slider_value_label
    
    root = tk.Tk()
    root.title("Traffic Sign Detection Interface")
    
    try:
        root.state('zoomed') 
    except tk.TclError:
        pass 
    
    root.config(bg=BACKGROUND_COLOR)
    
    # Configure the root grid to fill space for the main content area
    root.grid_rowconfigure(1, weight=1)
    root.grid_columnconfigure(0, weight=1)
    
    # --- 1. Control Panel (Top Bar) ---
    control_frame = tk.Frame(root, bg=BACKGROUND_COLOR)
    control_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
    
    # Set up grid inside the control_frame for horizontal layout
    control_frame.grid_columnconfigure(0, weight=0) # Button
    control_frame.grid_columnconfigure(1, weight=0) # Instruction Label
    control_frame.grid_columnconfigure(2, weight=1) # Spacer to push slider right
    control_frame.grid_columnconfigure(3, weight=0) # Slider title/control
    
    # Import Button
    import_button = tk.Button(
        control_frame, text="Import Image", command=import_image,
        font=("Arial", 10, "bold"), bg=BUTTON_COLOR, fg="#000000",
        borderwidth=0, width=15, height=1
    )
    import_button.grid(row=0, column=0, padx=(0, 20), pady=0)
    
    # Instruction Label (Used only initially)
    instruction_label = tk.Label(
        control_frame, text="To start the application, use the button --Import Image--",
        font=("Arial", 12), bg=BACKGROUND_COLOR, fg=TEXT_COLOR
    )
    instruction_label.grid(row=0, column=1, padx=(0, 20), pady=0, sticky="w")
    
    
    # --- Day/Night Threshold Slider and Labels (Right side of control frame) ---
    
    slider_container = tk.Frame(control_frame, bg=BACKGROUND_COLOR)
    slider_container.grid(row=0, column=3, sticky="e")
    
    slider_title_label = tk.Label(
        slider_container, text="Day/Night Auto Threshold (0-100):",
        font=("Arial", 10), bg=BACKGROUND_COLOR, fg=TEXT_COLOR
    )
    slider_title_label.grid(row=0, column=0, columnspan=2, sticky="w")

    threshold_slider = tk.Scale(
        slider_container, from_=0, to=100, orient=tk.HORIZONTAL, resolution=1, 
        command=update_threshold, 
        length=150, bg=BACKGROUND_COLOR, fg=TEXT_COLOR, 
        troughcolor=DISPLAY_FRAME_COLOR, highlightthickness=0,
        borderwidth=1, activebackground=BUTTON_COLOR,
        showvalue=0 
    )
    threshold_slider.set(THRESHOLD_MEAN)
    threshold_slider.grid(row=1, column=0, padx=(0, 5), sticky="w")
    
    slider_value_label = tk.Label(
        slider_container, text=f"Current Threshold: {THRESHOLD_MEAN}",
        font=("Arial", 10, "bold"), bg=BACKGROUND_COLOR, fg=TEXT_COLOR
    )
    slider_value_label.grid(row=1, column=1, sticky="w")


    # --- 2. Main Content Area (No Scrolling) ---
    
    main_content_frame = tk.Frame(root, bg=BACKGROUND_COLOR)
    main_content_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
    
    # Configure main content frame for 2 responsive columns
    main_content_frame.grid_columnconfigure(0, weight=1) # Left column (Images + Log)
    main_content_frame.grid_columnconfigure(1, weight=1) # Right column (Masks)
    
    
    # --- 2.1 Left Column: Images and Log ---
    left_column_frame = tk.Frame(main_content_frame, bg=BACKGROUND_COLOR)
    left_column_frame.grid(row=0, column=0, padx=(0, 10), sticky="n")
    
    # Frame for Original and Intermediate Images (Stacked)
    image_stack_frame = tk.Frame(left_column_frame, bg=BACKGROUND_COLOR)
    image_stack_frame.grid(row=0, column=0, sticky="n", pady=5)
    
    image_display_label = tk.Label(image_stack_frame, bg=DISPLAY_FRAME_COLOR, text="Original Image", 
                                     fg=TEXT_COLOR, font=("Arial", 12))
    image_display_label.pack(pady=5, padx=5) 
    
    intermediate_display_label = tk.Label(image_stack_frame, bg=DISPLAY_FRAME_COLOR, text="Intermediate Image", 
                                         fg=TEXT_COLOR, font=("Arial", 12))
    intermediate_display_label.pack(pady=10, padx=5) 

    # Detection Log (Text Box)
    results_label = tk.Label(left_column_frame, text="DETECTION LOG", font=("Arial", 12, "bold"), bg=BACKGROUND_COLOR, fg=TEXT_COLOR)
    results_label.grid(row=1, column=0, pady=(15, 5), sticky="w")
    
    # Log text box dimensions adjusted for fitting
    results_text = tk.Text(
        left_column_frame, height=8, width=50, wrap=tk.WORD,
        bg=DISPLAY_FRAME_COLOR, fg=TEXT_COLOR, font=("Consolas", 9), state=tk.DISABLED
    )
    results_text.grid(row=2, column=0, padx=5, pady=(0, 5), sticky="ew")
    
    
    # --- 2.2 Right Column: 3x3 Mask Grid ---
    
    mask_grid_frame = tk.Frame(main_content_frame, bg=BACKGROUND_COLOR)
    mask_grid_frame.grid(row=0, column=1, sticky="n") # Note: Sticky 'n' keeps it at the top
    
    
    root.mainloop()

if __name__ == "__main__":
    setup_gui()
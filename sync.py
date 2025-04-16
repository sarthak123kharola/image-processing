import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
from collections import Counter
import threading
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans


def visualize_brightness_distribution():
    global img
    if img is None:
        messagebox.showwarning("Warning", "No image loaded.")
        return
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    brightness_values = gray_img.flatten()
    
    sns.histplot(brightness_values, kde=True, color='gray')
    plt.xlabel('Brightness Value')
    plt.ylabel('Frequency')
    plt.title('Brightness Distribution')
    plt.show()

def visualize_rgb_channel_comparison():
    global img
    if img is None:
        messagebox.showwarning("Warning", "No image loaded.")
        return
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    colors = ('Red', 'Green', 'Blue')
    channels = cv2.split(img)
    
    for i, color in enumerate(colors):
        sns.histplot(channels[i].flatten(), kde=True, color=color.lower(), ax=axs[i])
        axs[i].set_title(f'{color} Channel Distribution')
        axs[i].set_xlabel('Pixel Value')
        axs[i].set_ylabel('Frequency')
    
    plt.show()

def visualize_color_correlation():
    global img
    if img is None:
        messagebox.showwarning("Warning", "No image loaded.")
        return
    
    r, g, b = cv2.split(img)
    flattened_data = np.array([r.flatten(), g.flatten(), b.flatten()])
    
    corr_matrix = np.corrcoef(flattened_data)
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', xticklabels=['Red', 'Green', 'Blue'], yticklabels=['Red', 'Green', 'Blue'])
    plt.title('RGB Channel Correlation Matrix')
    plt.show()

def visualize_pixel_intensity_across_regions():
    global img
    if img is None:
        messagebox.showwarning("Warning", "No image loaded.")
        return
    
    h, w, _ = img.shape
    regions = [
        ('Top Left', img[:h//2, :w//2]),
        ('Top Right', img[:h//2, w//2:]),
        ('Bottom Left', img[h//2:, :w//2]),
        ('Bottom Right', img[h//2:, w//2:])
    ]
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.ravel()
    
    for i, (region_name, region_img) in enumerate(regions):
        gray_region = cv2.cvtColor(region_img, cv2.COLOR_RGB2GRAY)
        sns.histplot(gray_region.flatten(), kde=True, color='gray', ax=axs[i])
        axs[i].set_title(f'Pixel Intensity in {region_name}')
        axs[i].set_xlabel('Pixel Value')
        axs[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

def visualize_dominant_colors(num_colors=5):
    global img
    if img is None:
        messagebox.showwarning("Warning", "No image loaded.")
        return
    
    try:
        if len(img.shape) == 3 and img.shape[2] == 3:
            pixels = img.reshape(-1, 3).astype(float)
        else:
            messagebox.showwarning("Warning", "Image format not supported for this function.")
            return
        
        clusters = KMeans(n_clusters=num_colors)
        clusters.fit(pixels)
        
        counts = Counter(clusters.labels_)
        center_colors = clusters.cluster_centers_
        
        ordered_colors = [center_colors[i] for i in counts.keys()]
        hex_colors = ['#%02x%02x%02x' % (int(color[0]), int(color[1]), int(color[2])) for color in ordered_colors]
    
        plt.figure(figsize=(8, 6))
        sns.barplot(x=list(counts.values()), y=hex_colors, palette=hex_colors)
        plt.xlabel('Pixel Count')
        plt.ylabel('Dominant Colors')
        plt.title(f'Top {num_colors} Dominant Colors')
        plt.show()

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

def visualize_histogram():
    global img
    if img is None:
        messagebox.showwarning("Warning", "No image loaded.")
        return
    
    channels = cv2.split(img)
    colors = ('r', 'g', 'b')
    
    plt.figure(figsize=(10, 6))
    
    for i, color in enumerate(colors):
        hist = cv2.calcHist([channels[i]], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        
        sns.lineplot(x=range(256), y=hist.flatten(), label=f'{color.upper()} channel', color=color)
        
        mean = np.mean(channels[i])
        std_dev = np.std(channels[i])
        plt.text(260, hist.max() * (0.9 - 0.1 * i), f'{color.upper()} Mean: {mean:.2f}', color=color)
        plt.text(260, hist.max() * (0.8 - 0.1 * i), f'{color.upper()} Std Dev: {std_dev:.2f}', color=color)
    
    plt.xlabel('Pixel Value')
    plt.ylabel('Normalized Frequency')
    plt.title('RGB Color Histogram and Statistics')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def initialize_mobilenet():
    return MobileNetV2(weights="imagenet")

def initialize_haar_cascade(path):
    return cv2.CascadeClassifier(path)

def initialize_yolo(weights_path, cfg_path, names_path):
    net = cv2.dnn.readNet(weights_path, cfg_path)
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

model = initialize_mobilenet()
face_cascade = initialize_haar_cascade('D:\image processing\haarcascade_frontalface_default.xml')
net, classes, output_layers = initialize_yolo(
    'D:\image processing\yolov3.weights',
    'D:\image processing\yolov3.cfg',
    'D:\image processing\coco.names'
)

img = None
original_img = None
processed_img = None
image_history = []
history_index = -1

def calculate_brightness(image):
    return np.mean(image)

def color_histogram(image):
    histogram = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    histogram = cv2.normalize(histogram, histogram).flatten()
    return histogram

def dominant_colors(image, num_colors=3):
    pixels = image.reshape(-1, 3)
    colors = Counter(map(tuple, pixels)).most_common(num_colors)
    return colors

def classify_image():
    global img
    if img is None:
        messagebox.showwarning("Warning", "No image loaded.")
        return

    image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    
    predictions = model.predict(image)
    label = decode_predictions(predictions, top=1)[0][0]
    class_name = label[1]
    confidence = label[2]
    
    width, height = img.shape[1], img.shape[0]
    avg_brightness = calculate_brightness(img)
    histogram = color_histogram(img)
    total_pixels = width * height
    aspect_ratio = width / height
    dominant_colors_info = dominant_colors(img)

    result_text = f"### Image Information\n" \
                  f"- **Class**: {class_name}\n" \
                  f"- **Confidence**: {confidence:.2f}\n" \
                  f"- **Dimensions**: {width}x{height}\n" \
                  f"- **Total Pixels**: {total_pixels}\n" \
                  f"- **Aspect Ratio**: {aspect_ratio:.2f}\n" \
                  f"- **Average Brightness**: {avg_brightness:.2f}\n" \
                  f"- **Color Histogram**: {histogram.tolist()}\n" \
                  f"- **Dominant Colors**: {dominant_colors_info}\n"
                  
    update_info_box(result_text)

def detect_faces():
    global img, processed_img
    if img is None:
        messagebox.showwarning("Warning", "No image loaded.")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 3)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    processed_img = img.copy()
    add_to_history(processed_img)
    show_image(processed_img)
    update_info_box(f"### Detected Faces\n- **Count**: {len(faces)}")

def detect_objects():
    global img, processed_img
    if img is None:
        messagebox.showwarning("Warning", "No image loaded.")
        return
    
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 1/255, (608, 608), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids, confidences, boxes = [], [], []
    detected_objects = []
    count=0
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.6)
    
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f"{label} {confidence:.2f}", (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, color, 3)
            detected_objects.append(f"{label} ({confidence:.2f})")
            count+=1
    
    processed_img = img.copy()
    add_to_history(processed_img)
    show_image(processed_img)
    update_info_box(f"### Total detected Objects - {count}\n### Detected Objects\n- " + "\n- ".join(detected_objects))


def add_to_history(image):
    global history_index
    image_history[:] = image_history[:history_index + 1]
    image_history.append(image)
    history_index += 1

def undo_last_action():
    global img, processed_img, history_index
    if history_index > 0:
        history_index -= 1
        processed_img = image_history[history_index]
        show_image(processed_img)
        update_info_box("Undid last action.")
    else:
        messagebox.showwarning("Warning", "No action to undo.")

def redo_last_action():
    global img, processed_img, history_index
    if history_index==-1 or history_index>=len(image_history)-1:
        messagebox.showwarning("Warning","No action to redo.")
    else:
        history_index += 1
        processed_img=image_history[history_index]
        show_image(processed_img)
        update_info_box("Redid last action.")
        
def open_file():
    global img, original_img, processed_img, image_history, history_index
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)
        if img is None:
            messagebox.showerror("Error", "Could not open image.")
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_img = img.copy()
        processed_img = img.copy()
        add_to_history(processed_img)
        show_image(img)
        update_info_box("Image loaded successfully.")

def reset_image():
    global img, original_img, processed_img
    if original_img is not None:
        img = original_img.copy()
        processed_img = original_img.copy()
        add_to_history(processed_img)
        show_image(img)
        update_info_box("Image reset to original state.")
    else:
        messagebox.showwarning("Warning", "No original image to reset to.")

def save_image():
    global processed_img
    if processed_img is None:
        messagebox.showwarning("Warning", "No image to save.")
        return
    
    save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")])
    if save_path:
        img_to_save = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, img_to_save)
        update_info_box("Image saved successfully.")

def apply_filter(filter_type):

    global img, processed_img
    temp= img
    img= processed_img

    if img is None:
        messagebox.showwarning("Warning", "No image loaded.")
        return

    if filter_type == 'Blur':
        processed_img = cv2.GaussianBlur(img, (15, 15), 0)
    elif filter_type == 'Sharpen':
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        processed_img = cv2.filter2D(img, -1, kernel)
    elif filter_type == 'Grayscale':
        processed_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif filter_type == 'Invert':
        processed_img = cv2.bitwise_not(img)
    elif filter_type == 'Canny':
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        processed_img = cv2.Canny(gray, 100, 200)
    elif filter_type == 'Rotate':
        processed_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif filter_type == 'Brightness':
        beta = 50 
        processed_img = cv2.convertScaleAbs(img, beta=beta)
    elif filter_type == 'Shear':
        rows, cols, _ = img.shape
        M = np.float32([[1, 0.5, 0], [0, 1, 0]])
        processed_img = cv2.warpAffine(img, M, (cols, rows))
    elif filter_type == 'Translate':
        rows, cols, _ = img.shape
        M = np.float32([[1, 0, 50], [0, 1, 50]])
        processed_img = cv2.warpAffine(img, M, (cols, rows))
    elif filter_type == 'Sepia':
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                  [0.349, 0.686, 0.168],
                                  [0.393, 0.769, 0.189]])
        processed_img = cv2.transform(img, sepia_filter)
        processed_img = np.clip(processed_img, 0, 255).astype(np.uint8)
    elif filter_type == 'Edge Detection':
        processed_img = cv2.Canny(img, 100, 200)

    img=temp

    add_to_history(processed_img)
    show_image(processed_img)
    update_info_box(f"{filter_type} filter applied.")

def show_image(image):
    if len(image.shape) == 2: 
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = Image.fromarray(image)

    image.thumbnail((canvas.winfo_width(), canvas.winfo_height()), Image.LANCZOS)
    image = ImageTk.PhotoImage(image)

    canvas.create_image(0, 0, anchor=tk.NW, image=image)
    canvas.image = image

def _on_mouse_wheel(event):
    if info_label.winfo_containing(event.x_root, event.y_root) is info_label:
        info_label.yview_scroll(int(-1*(event.delta/120)), "units")

def update_info_box(info_text):
    info_label.config(state=tk.NORMAL)
    info_label.delete(1.0, tk.END)
    info_label.insert(tk.END, info_text)
    info_label.config(state=tk.DISABLED)
    info_label.yview(tk.END)

def start_live_feed():
    global is_live_camera
    is_live_camera = True
    threading.Thread(target=update_live_feed).start()

def update_live_feed():
    global img, processed_img, is_live_camera
    cap = cv2.VideoCapture(0)
    while is_live_camera:
        ret, frame = cap.read()
        if not ret:
            break
        
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_img = img.copy()
        
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        show_image(img)
        root.update_idletasks()
        root.update()

    cap.release()

def stop_live_feed():
    global is_live_camera
    is_live_camera = False

root = tk.Tk()
root.title("Enhanced Image Processing Application")
root.geometry("1200x700")
root.configure(bg='#2a2a72')

style = ttk.Style()
style.configure('TButton', font=('Helvetica', 12), padding=15)
style.map('TButton',
          foreground=[('!disabled', 'black')],
          background=[('!disabled', '#00ccff'), ('active', '#0099cc')],
          relief=[('pressed', 'groove'), ('!pressed', 'ridge')])

canvas = tk.Canvas(root, width=800, height=600, bg='#ffffff', borderwidth=2, relief='sunken')
canvas.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)

info_label = tk.Text(root, bg='#2a2a72', fg='white', font=('Helvetica', 14), wrap=tk.WORD, height=15, width=50)
info_label.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
info_label.config(state=tk.DISABLED)

frame = tk.Frame(root, bg='#2a2a72')
frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)

canvas_frame = tk.Canvas(frame, bg='#2a2a72')
scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas_frame.yview)
scrollable_frame = tk.Frame(canvas_frame, bg='#2a2a72')

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas_frame.configure(
        scrollregion=canvas_frame.bbox("all")
    )
)

canvas_frame.create_window((0, 0), window=scrollable_frame, anchor=tk.NW)
canvas_frame.configure(yscrollcommand=scrollbar.set)

canvas_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

operations = [
    ("Visualize", visualize_histogram),
    ("Visualize Brightness Distribution", visualize_brightness_distribution),
    ("Visualize Color Correlation", visualize_color_correlation),
    ("Visualize Dominant Colors", lambda: visualize_dominant_colors(4)),
    ("Visualize Pixel Intensity", visualize_pixel_intensity_across_regions),
    ("Visualize Rgb Channel Comparision", visualize_rgb_channel_comparison),
    ("Open Image", open_file),
    ("Reset Image", reset_image),
    ("Save Image", save_image),
    ("Undo Last Action", undo_last_action),
    ("Redo Action", redo_last_action),
    ("Classify Image", classify_image),
    ("Detect Faces", detect_faces),
    ("Detect Objects", detect_objects),
    ("Apply Blur Filter", lambda: apply_filter('Blur')),
    ("Apply Sharpen Filter", lambda: apply_filter('Sharpen')),
    ("Apply Grayscale Filter", lambda: apply_filter('Grayscale')),
    ("Apply Invert Filter", lambda: apply_filter('Invert')),
    ("Apply Canny Edge Detection", lambda: apply_filter('Canny')),
    ("Rotate Image", lambda: apply_filter('Rotate')),
    ("Adjust Brightness", lambda: apply_filter('Brightness')),
    ("Apply Shear", lambda: apply_filter('Shear')),
    ("Apply Translation", lambda: apply_filter('Translate')),
    ("Apply Sepia Filter", lambda: apply_filter('Sepia')),
    ("Apply Edge Detection", lambda: apply_filter('Edge Detection')),
    ("Start Live Feed", start_live_feed),
    ("Stop Live Feed", stop_live_feed)
]

for (text, command) in operations:
    ttk.Button(scrollable_frame, text=text, command=command, width=25).pack(side=tk.TOP, padx=10, pady=5, fill=tk.X)

status_bar = tk.Label(root, text="Welcome to the Image Processing Application!", bg='#2a2a72', fg='white', anchor='w')
status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10)

info_label.bind("<Enter>", lambda e: root.bind("<MouseWheel>", _on_mouse_wheel))
info_label.bind("<Leave>", lambda e: root.unbind("<MouseWheel>"))

root.mainloop()
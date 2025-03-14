# import sys
# import cv2
# import numpy as np
# import os

# # Suppress terminal on Windows (for pythonw.exe)
# if os.name == "nt":
#     import ctypes
#     ctypes.windll.kernel32.FreeConsole()

# # Get video file path from arguments or use default
# video_path = sys.argv[1] if len(sys.argv) > 1 else "traffic_video.mp4"

# # Check if file exists
# if not os.path.exists(video_path):
#     print(f"Error: Video file '{video_path}' does not exist! Please check the path.")
#     sys.exit(1)

# # Open the video file
# cap = cv2.VideoCapture(video_path)
# if not cap.isOpened():
#     print(f"Error: Unable to open video file {video_path}")
#     sys.exit(1)

# # Reduce resolution for faster processing
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# cap.set(cv2.CAP_PROP_FPS, 30)

# # Use KNN background subtractor for better performance
# fgbg = cv2.createBackgroundSubtractorKNN()

# # Initialize heatmap accumulator
# heatmap_accumulator = None

# # Create OpenCV window
# cv2.namedWindow("Traffic Density Heatmap", cv2.WINDOW_NORMAL)
# cv2.setWindowProperty("Traffic Density Heatmap", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break  # Exit loop if video ends

#     # Convert frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Apply background subtraction
#     fgmask = fgbg.apply(gray)

#     # Threshold to remove noise
#     _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

#     # Initialize heatmap accumulator
#     if heatmap_accumulator is None:
#         heatmap_accumulator = np.zeros_like(thresh, dtype=np.float32)

#     # Decay previous heatmap values slightly
#     heatmap_accumulator *= 0.85
#     heatmap_accumulator += thresh

#     # Normalize the heatmap
#     heatmap_norm = cv2.normalize(heatmap_accumulator, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

#     # Apply colormap to visualize congestion
#     heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

#     # Overlay heatmap onto the original frame
#     overlay = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)

#     # Display using OpenCV
#     cv2.imshow("Traffic Density Heatmap", overlay)

#     # Exit on 'q' or ESC key
#     key = cv2.waitKey(1)
#     if key in [ord('q'), 27]:  # 27 is ESC key
#         break

# cap.release()
# cv2.destroyAllWindows()

# print("Traffic density visualization completed successfully!")



import sys
import cv2
import numpy as np
from ultralytics import YOLO
import os

# ✅ Suppress terminal on Windows
if os.name == "nt":
    import ctypes
    ctypes.windll.kernel32.FreeConsole()

# ✅ Load YOLOv8 Model
model = YOLO("yolov8n.pt")  # Pre-trained YOLOv8 model

# ✅ Get video path from CLI or use default
video_path = sys.argv[1] if len(sys.argv) > 1 else "test_video.mp4"

def generate_heatmap(frame, vehicle_positions, intensity=15):
    """Generate a heatmap of vehicle positions and overlay it on the frame."""
    heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
    
    for x, y in vehicle_positions:
        heatmap[int(y), int(x)] += 1
    
    heatmap = cv2.GaussianBlur(heatmap, (0, 0), intensity)
    heatmap = np.clip(heatmap / heatmap.max(), 0, 1) * 255  # Normalize

    heatmap_colored = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 0.6, heatmap_colored, 0.4, 0)

def process_frame(frame, model, prev_positions):
    """Process a video frame: Detect objects, generate heatmap, and overlay information."""
    results = model(frame)
    detections = results[0].boxes.data.cpu().numpy()
    vehicle_positions = []

    for det in detections:
        x1, y1, x2, y2, conf, class_id = det
        class_id = int(class_id)
        label = model.names[class_id]
        
        if label in ["car", "truck", "motorcycle", "bus"]:  # Only count vehicles
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            vehicle_positions.append((center_x, center_y))

            color = (0, 255, 0) if conf > 0.5 else (0, 0, 255)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # ✅ Add Heatmap Overlay
    frame = generate_heatmap(frame, vehicle_positions)

    return frame, vehicle_positions

def process_video(video_path):
    """Process a single video and display results in full screen."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Error: Unable to open video file {video_path}")
        return

    prev_positions = []

    # ✅ Open full-screen OpenCV window
    cv2.namedWindow("Traffic Heatmap Visualization", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Traffic Heatmap Visualization", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, vehicle_positions = process_frame(frame, model, prev_positions)
        prev_positions = vehicle_positions

        cv2.imshow("Traffic Heatmap Visualization", frame)

        # ✅ Press 'q' or ESC to exit
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:  # 27 is ESC key
            break

    # ✅ **Hold video at end until manual close**
    while True:
        key = cv2.waitKey(0)
        if key == ord('q') or key == 27:
            break

    print("✅ Video finished. Press any key to exit.")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video(video_path)

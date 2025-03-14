# import sys
# import cv2
# import numpy as np
# from ultralytics import YOLO  # Import YOLOv8

# # Constants
# FRAME_RATE = 30
# MOTION_THRESHOLD = 20
# ACCIDENT_AREA_THRESHOLD = 500
# SKIP_FRAMES = 2
# RESIZE_FACTOR = 0.5

# # Load YOLO model
# model = YOLO("yolov8n.pt")  # Use the YOLOv8 model

# # Check if a video file path is provided as an argument; otherwise, use a default video.
# if len(sys.argv) > 1:
#     video_path = sys.argv[1]
# else:
#     video_path = "Accident_Videos.mp4"  # Default video for testing

# # Open the video file
# video = cv2.VideoCapture(video_path)
# if not video.isOpened():
#     print(f"Error: Unable to open video file {video_path}")
#     sys.exit(1)  # Exit with error code

# vehicle_tracks = {}
# accident_detected = False

# # Function to calculate overlap between two bounding boxes
# def calculate_overlap(box1, box2):
#     x1 = max(box1[0], box2[0])
#     y1 = max(box1[1], box2[1])
#     x2 = min(box1[0] + box1[2], box2[0] + box2[2])
#     y2 = min(box1[1] + box1[3], box2[1] + box2[3])
#     if x2 < x1 or y2 < y1:
#         return 0
#     return (x2 - x1) * (y2 - y1)

# # Function to draw a traffic signal on the screen
# def draw_traffic_signal(frame, signal_color):
#     signal_box_x = 10
#     signal_box_y = 10
#     signal_box_width = 60
#     signal_box_height = 160

#     # Draw signal box
#     cv2.rectangle(frame, (signal_box_x, signal_box_y), 
#                   (signal_box_x + signal_box_width, signal_box_y + signal_box_height), 
#                   (0, 0, 0), -1)
#     cv2.rectangle(frame, (signal_box_x, signal_box_y), 
#                   (signal_box_x + signal_box_width, signal_box_y + signal_box_height), 
#                   (255, 255, 255), 2)

#     # Draw red, yellow, and green lights
#     cv2.circle(frame, (signal_box_x + 30, signal_box_y + 30), 15, 
#                (0, 0, 255) if signal_color == "red" else (50, 50, 50), -1)
#     cv2.circle(frame, (signal_box_x + 30, signal_box_y + 80), 15, 
#                (0, 255, 255) if signal_color == "yellow" else (50, 50, 50), -1)
#     cv2.circle(frame, (signal_box_x + 30, signal_box_y + 130), 15, 
#                (0, 255, 0) if signal_color == "green" else (50, 50, 50), -1)

# # Create a resizable OpenCV window
# cv2.namedWindow("Accident Detection", cv2.WINDOW_NORMAL)
# cv2.setWindowProperty("Accident Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# frame_id = 0
# while True:
#     ret, frame = video.read()
#     if not ret:
#         print("End of video or error reading frame.")
#         break

#     frame_id += 1

#     if frame_id % SKIP_FRAMES != 0:
#         continue

#     original_frame = frame.copy()

#     height, width = frame.shape[:2]
#     resized_frame = cv2.resize(frame, (int(width * RESIZE_FACTOR), int(height * RESIZE_FACTOR)))

#     # Perform object detection using YOLO
#     results = model(resized_frame, conf=0.5)

#     boxes = []
#     confidences = []
#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
#             confidence = box.conf[0].cpu().numpy()

#             x, y, w, h = int(x1 / RESIZE_FACTOR), int(y1 / RESIZE_FACTOR), int((x2 - x1) / RESIZE_FACTOR), int((y2 - y1) / RESIZE_FACTOR)
#             boxes.append([x, y, w, h])
#             confidences.append(float(confidence))

#     if len(boxes) > 0 and len(confidences) > 0:
#         indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
#     else:
#         indices = []

#     current_frame_tracks = {}
#     for i in indices:
#         i = i
#         box = boxes[i]
#         x, y, w, h = box

#         vehicle_id = f"{x}_{y}"
#         current_frame_tracks[vehicle_id] = box

#         # Check for overlapping objects (possible accident detection)
#         for other_id, other_box in current_frame_tracks.items():
#             if vehicle_id != other_id:
#                 overlap_area = calculate_overlap(box, other_box)
#                 if overlap_area > ACCIDENT_AREA_THRESHOLD:
#                     accident_detected = True
#                     cv2.putText(original_frame, "Accident Detected!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#                     cv2.rectangle(original_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

#         cv2.rectangle(original_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#     # Display traffic signal based on accident detection
#     signal_color = "red" if accident_detected else "green"
#     draw_traffic_signal(original_frame, signal_color)

#     # Show the frame with detections
#     cv2.imshow("Accident Detection", original_frame)

#     # Reset accident detection flag
#     accident_detected = False

#     # Press 'q' to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# video.release()
# cv2.destroyAllWindows()


import sys
import cv2
import numpy as np
import os
from ultralytics import YOLO

# Suppress terminal on Windows (for pythonw.exe)
if os.name == "nt":
    import ctypes
    ctypes.windll.kernel32.FreeConsole()

# Constants
ACCIDENT_AREA_THRESHOLD = 500
SKIP_FRAMES = 2
RESIZE_FACTOR = 0.5

# Load YOLO model
model = YOLO("yolov8n.pt")  # Use the YOLOv8 model

# Check if a video file path is provided as an argument; otherwise, use a default video.
video_path = sys.argv[1] if len(sys.argv) > 1 else "Accident_Videos.mp4"

# Open the video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Unable to open video file {video_path}")
    sys.exit(1)  # Exit with error code

# Function to calculate overlap between two bounding boxes
def calculate_overlap(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    if x2 < x1 or y2 < y1:
        return 0
    return (x2 - x1) * (y2 - y1)

# Function to draw a traffic signal on the screen
def draw_traffic_signal(frame, signal_color):
    signal_box_x, signal_box_y = 10, 10
    signal_width, signal_height = 60, 160

    # Draw signal box
    cv2.rectangle(frame, (signal_box_x, signal_box_y), 
                  (signal_box_x + signal_width, signal_box_y + signal_height), 
                  (0, 0, 0), -1)
    cv2.rectangle(frame, (signal_box_x, signal_box_y), 
                  (signal_box_x + signal_width, signal_box_y + signal_height), 
                  (255, 255, 255), 2)

    # Draw red, yellow, and green lights
    cv2.circle(frame, (signal_box_x + 30, signal_box_y + 30), 15, 
               (0, 0, 255) if signal_color == "red" else (50, 50, 50), -1)
    cv2.circle(frame, (signal_box_x + 30, signal_box_y + 80), 15, 
               (0, 255, 255) if signal_color == "yellow" else (50, 50, 50), -1)
    cv2.circle(frame, (signal_box_x + 30, signal_box_y + 130), 15, 
               (0, 255, 0) if signal_color == "green" else (50, 50, 50), -1)

# Create a resizable OpenCV window
cv2.namedWindow("Accident Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Accident Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

frame_id = 0
accident_detected = False

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if video ends

    frame_id += 1
    if frame_id % SKIP_FRAMES != 0:
        continue

    original_frame = frame.copy()
    height, width = frame.shape[:2]
    resized_frame = cv2.resize(frame, (int(width * RESIZE_FACTOR), int(height * RESIZE_FACTOR)))

    # Perform object detection using YOLO
    results = model(resized_frame, conf=0.5)

    boxes, confidences = [], []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].cpu().numpy()

            x, y, w, h = int(x1 / RESIZE_FACTOR), int(y1 / RESIZE_FACTOR), int((x2 - x1) / RESIZE_FACTOR), int((y2 - y1) / RESIZE_FACTOR)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3) if len(boxes) > 0 else []

    current_frame_tracks = {}
    for i in indices:
        i = i
        box = boxes[i]
        x, y, w, h = box

        vehicle_id = f"{x}_{y}"
        current_frame_tracks[vehicle_id] = box

        # Check for overlapping objects (possible accident detection)
        for other_id, other_box in current_frame_tracks.items():
            if vehicle_id != other_id:
                overlap_area = calculate_overlap(box, other_box)
                if overlap_area > ACCIDENT_AREA_THRESHOLD:
                    accident_detected = True
                    cv2.putText(original_frame, "Accident Detected!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.rectangle(original_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.rectangle(original_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display traffic signal based on accident detection
    signal_color = "red" if accident_detected else "green"
    draw_traffic_signal(original_frame, signal_color)

    # Show the frame with detections
    cv2.imshow("Accident Detection", original_frame)

    # Reset accident detection flag
    accident_detected = False

    # Press 'q' or ESC to exit
    key = cv2.waitKey(1)
    if key in [ord('q'), 27]:  # 27 is ESC key
        break

cap.release()
cv2.destroyAllWindows()

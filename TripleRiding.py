# import sys
# import cv2
# from ultralytics import YOLO  # Import YOLOv8

# # Load YOLOv8 model
# model = YOLO('yolov8s.pt')  # 'yolov8s.pt' is the small version; replace if needed

# # Set classes for motorbike and person (COCO dataset IDs)
# motorbike_class = 3  # Class ID for 'motorbike'
# person_class = 0     # Class ID for 'person'

# # Check if a video file path is provided as an argument; otherwise, use a default video.
# if len(sys.argv) > 1:
#     video_path = sys.argv[1]
# else:
#     video_path = "bike_3.mp4"  # Default video for testing

# # Open the video file
# cap = cv2.VideoCapture(video_path)
# if not cap.isOpened():
#     print(f"Error: Unable to open video file {video_path}")
#     sys.exit(1)  # Exit with error code

# # Create a resizable OpenCV window
# cv2.namedWindow("Triple Riding Detection", cv2.WINDOW_NORMAL)
# cv2.setWindowProperty("Triple Riding Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Perform object detection on the frame
#     results = model(frame)
#     detections = results[0].boxes.data.cpu().numpy()  # Convert to NumPy array

#     motorbikes = []
#     people = []

#     # Separate detections for motorbike and person
#     for detection in detections:
#         x1, y1, x2, y2, score, class_id = detection
#         if int(class_id) == motorbike_class:
#             motorbikes.append([x1, y1, x2, y2])
#         elif int(class_id) == person_class:
#             people.append([x1, y1, x2, y2])

#     # Analyze each motorbike for triple riding
#     for i, bike in enumerate(motorbikes, start=1):
#         x1_b, y1_b, x2_b, y2_b = bike
#         person_count = 0

#         for person in people:
#             x1_p, y1_p, x2_p, y2_p = person
#             # Simple overlap check (could be replaced with IoU calculation)
#             if (x1_p < x2_b and x2_p > x1_b and y1_p < y2_b and y2_p > y1_b):
#                 person_count += 1

#         status = "offense" if person_count >= 3 else "not offense"
#         print(f"Motorbike {i}: people ({status})")

#         # Annotate the frame
#         color = (0, 0, 255) if status == "offense" else (0, 255, 0)
#         cv2.rectangle(frame, (int(x1_b), int(y1_b)), (int(x2_b), int(y2_b)), color, 2)
#         cv2.putText(frame, f"({status})", (int(x1_b), int(y1_b) - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#     # Display the frame
#     cv2.imshow("Triple Riding Detection", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


import sys
import os
import cv2
import subprocess
from ultralytics import YOLO  # Import YOLOv8

# --- Step 1: Relaunch Script in the Background if Needed ---
if os.name == "nt" and not sys.stdout:  # Check if running in console
    DETACHED_PROCESS = 0x00000008
    subprocess.Popen(["pythonw", sys.argv[0]] + sys.argv[1:], creationflags=DETACHED_PROCESS)
    sys.exit()  # Exit the current script so only the background process runs

# --- Load YOLO Model ---
model = YOLO('yolov8s.pt')

# --- Set Classes for Motorbike and Person ---
motorbike_class = 3  # Class ID for 'motorbike'
person_class = 0     # Class ID for 'person'

# --- Video Input ---
video_path = sys.argv[1] if len(sys.argv) > 1 else "bike_3.mp4"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Unable to open video file {video_path}")
    sys.exit(1)  # Exit with error code

# --- Create a Resizable OpenCV Window ---
cv2.namedWindow("Triple Riding Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Triple Riding Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Perform Object Detection ---
    results = model(frame)
    detections = results[0].boxes.data.cpu().numpy()

    motorbikes, people = [], []

    for detection in detections:
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) == motorbike_class:
            motorbikes.append([x1, y1, x2, y2])
        elif int(class_id) == person_class:
            people.append([x1, y1, x2, y2])

    for i, bike in enumerate(motorbikes, start=1):
        x1_b, y1_b, x2_b, y2_b = bike
        person_count = sum(
            x1_p < x2_b and x2_p > x1_b and y1_p < y2_b and y2_p > y1_b
            for x1_p, y1_p, x2_p, y2_p in people
        )

        status = "offense" if person_count >= 3 else "not offense"
        print(f"Motorbike {i}: {person_count} people ({status})")

        color = (0, 0, 255) if status == "offense" else (0, 255, 0)
        cv2.rectangle(frame, (int(x1_b), int(y1_b)), (int(x2_b), int(y2_b)), color, 2)
        cv2.putText(frame, f"{person_count} ({status})", (int(x1_b), int(y1_b) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # --- Display the Frame ---
    cv2.imshow("Triple Riding Detection", frame)
    
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:  # Exit on 'q' or 'ESC'
        break

cap.release()
cv2.destroyAllWindows()

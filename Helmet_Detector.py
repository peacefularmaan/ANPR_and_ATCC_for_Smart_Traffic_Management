# import sys
# import cv2
# import math
# import cvzone
# from ultralytics import YOLO

# # Check if a video file path is provided as an argument; otherwise, use default.
# if len(sys.argv) > 1:
#     video_path = sys.argv[1]
# else:
#     video_path = "bike_3.mp4"  # Default video for testing

# # Open the video file
# cap = cv2.VideoCapture(video_path)
# if not cap.isOpened():
#     print(f"Error: Unable to open video file {video_path}")
#     sys.exit(1)  # Exit with error code

# # Load YOLO model with custom weights
# model = YOLO("Weights/best.pt")

# # Define class names (modify as per your dataset)
# classNames = ['', '']

# frame_skip = 8  # Process every 8th frame
# frame_count = 0

# # Desired output dimensions
# output_width, output_height = 640, 480

# # Create a resizable OpenCV window
# cv2.namedWindow("Helmet Detection", cv2.WINDOW_NORMAL)
# cv2.setWindowProperty("Helmet Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# while True:
#     success, img = cap.read()
#     if not success:
#         break

#     # Resize frame to fit the output window completely
#     img = cv2.resize(img, (output_width, output_height))

#     frame_count += 1
#     if frame_count % frame_skip != 0:
#         continue  # Skip frame processing

#     results = model(img, stream=True)
#     for r in results:
#         boxes = r.boxes
#         for box in boxes:
#             x1, y1, x2, y2 = box.xyxy[0]
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#             w, h = x2 - x1, y2 - y1

#             # Set border color based on detection class
#             border_color = (0, 255, 0) if int(box.cls[0]) == 0 else (0, 0, 255)  # Green for helmet, Red for no helmet
#             cvzone.cornerRect(img, (x1, y1, w, h), colorC=border_color)

#             conf = math.ceil((box.conf[0] * 100)) / 100
#             cls = int(box.cls[0])
#             cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

#     cv2.imshow("Helmet Detection", img)

#     # Press 'q' to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


import sys
import cv2
import math
import cvzone
from ultralytics import YOLO
import os

# Suppress terminal on Windows (for pythonw.exe)
if os.name == "nt":
    import ctypes
    ctypes.windll.kernel32.FreeConsole()

# Check if a video file path is provided as an argument; otherwise, use default.
video_path = sys.argv[1] if len(sys.argv) > 1 else "bike_3.mp4"

# Open the video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Unable to open video file {video_path}")
    sys.exit(1)  # Exit with error code

# Load YOLO model with custom weights
model = YOLO("Weights/best.pt")

# Define class names (modify as per your dataset)
classNames = ['', '']

frame_skip = 8  # Process every 8th frame
frame_count = 0

# Desired output dimensions
output_width, output_height = 640, 480

# Create a resizable OpenCV window
cv2.namedWindow("Helmet Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Helmet Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    success, img = cap.read()
    if not success:
        break

    # Resize frame to fit the output window completely
    img = cv2.resize(img, (output_width, output_height))

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  # Skip frame processing

    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Set border color based on detection class
            border_color = (0, 255, 0) if int(box.cls[0]) == 0 else (0, 0, 255)  # Green for helmet, Red for no helmet
            cvzone.cornerRect(img, (x1, y1, w, h), colorC=border_color)

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    cv2.imshow("Helmet Detection", img)

    # Press 'q' to exit the OpenCV window
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:  # 27 is ESC key
        break

cap.release()
cv2.destroyAllWindows()

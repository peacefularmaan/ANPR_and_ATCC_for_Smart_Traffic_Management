# import sys
# import cv2
# import numpy as np
# import time
# import os
# from ultralytics import YOLO

# # Load the YOLOv8 model
# model = YOLO("yolov8n.pt")  

# # COCO dataset vehicle class IDs: Car=2, Motorcycle=3, Bus=5, Truck=7
# coco_classes = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# # Check if video file paths are provided as arguments; otherwise, use default videos.
# if len(sys.argv) > 1:
#     video_paths = sys.argv[1:]  # Take multiple video files from arguments
# else:
#     video_paths = [
#         "ATCC1.mp4",  
#         "ATCC2.mp4",  
#         "ATCC3.mp4",
#         "ATCC4.mp4"
#     ]  

# # Check if all files exist
# valid_paths = [path for path in video_paths if os.path.exists(path)]
# if not valid_paths:
#     print("Error: No valid video files found! Please check the file paths.")
#     sys.exit(1)

# # Open video captures for available paths
# caps = [cv2.VideoCapture(path) for path in valid_paths]

# frame_width, frame_height = 640, 360

# # Initialize vehicle tracking variables
# vehicle_counts = [0] * len(caps)
# vehicle_classifications = [{} for _ in range(len(caps))]

# # Create OpenCV window
# cv2.namedWindow("Traffic Management System", cv2.WINDOW_NORMAL)
# cv2.setWindowProperty("Traffic Management System", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# while any(cap.isOpened() for cap in caps):
#     frames = []
#     total_classifications = {c: 0 for c in coco_classes.values()}
#     total_vehicle_count = 0

#     for i, cap in enumerate(caps):
#         ret, frame = cap.read()
#         if not ret:
#             frames.append(np.zeros((frame_height, frame_width, 3), dtype=np.uint8))  # Empty frame
#             continue

#         frame = cv2.resize(frame, (frame_width, frame_height))
#         results = model(frame)

#         vehicle_count = 0
#         classifications = {}

#         for obj in results[0].boxes.data:
#             x1, y1, x2, y2, confidence, class_id = obj[:6]
#             x1, y1, x2, y2, class_id = map(int, [x1, y1, x2, y2, class_id])
#             if class_id in coco_classes and confidence > 0.5:
#                 vehicle_count += 1
#                 vehicle_type = coco_classes[class_id]
#                 classifications[vehicle_type] = classifications.get(vehicle_type, 0) + 1
#                 total_classifications[vehicle_type] += 1
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, f"{vehicle_type} {confidence:.2f}", (x1, y1 - 5),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         vehicle_counts[i] = vehicle_count
#         vehicle_classifications[i] = classifications
#         total_vehicle_count += vehicle_count

#         # Display vehicle count on each video
#         cv2.putText(frame, f"Count: {vehicle_count}", (20, 40),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

#         frames.append(frame)

#     # Determine the busiest road
#     max_index = np.argmax(vehicle_counts)

#     for i, frame in enumerate(frames):
#         red_color = (0, 0, 255) if i != max_index else (255, 255, 255)  # Red light for non-busiest roads
#         green_color = (0, 255, 0) if i == max_index else (255, 255, 255)  # Green light for busiest road

#         # Draw traffic signals
#         cv2.circle(frame, (50, 80), 15, red_color, -1)  # Red light
#         cv2.circle(frame, (50, 110), 15, green_color, -1)  # Green light

#     # Create a summary panel at the top
#     summary_frame = np.zeros((70, frame_width * len(frames), 3), dtype=np.uint8)
#     total_count_text = f"Total Vehicles: {total_vehicle_count}"
#     class_text = " | ".join([f"{v}: {total_classifications.get(v, 0)}" for v in coco_classes.values()])
#     cv2.putText(summary_frame, total_count_text, (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#     cv2.putText(summary_frame, class_text, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

#     # Stack video feeds dynamically
#     if len(frames) == 1:
#         final_display = frames[0]
#     elif len(frames) == 2:
#         final_display = np.hstack(frames)
#     else:
#         mid = len(frames) // 2
#         top_row = np.hstack(frames[:mid])
#         bottom_row = np.hstack(frames[mid:]) if len(frames) > mid else np.zeros_like(top_row)
#         final_display = np.vstack([top_row, bottom_row])

#     # Resize summary panel to match video width
#     summary_frame = cv2.resize(summary_frame, (final_display.shape[1], 70))
#     final_display = np.vstack([summary_frame, final_display])

#     cv2.imshow("Traffic Management System", final_display)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# for cap in caps:
#     cap.release()
# cv2.destroyAllWindows()


# import cv2
# import numpy as np
# from ultralytics import YOLO

# # Load YOLO model
# model = YOLO("yolov8n.pt")

# # Video paths
# video_paths = [
#     "car1.mp4",  # North
#     "car8.mp4",  # South
#     "car7.mp4",  # East
#     "car4.mp4"   # West
# ]

# # Open video files
# caps = [cv2.VideoCapture(vp) if cv2.VideoCapture(vp).isOpened() else None for vp in video_paths]

# # Frame size
# FRAME_WIDTH, FRAME_HEIGHT = 320, 240

# # Traffic light states
# traffic_lights = {"North": "RED", "South": "RED", "East": "RED", "West": "RED"}
# green_timers = {"North": 0, "South": 0, "East": 0, "West": 0}
# yellow_timers = {"North": 0, "South": 0, "East": 0, "West": 0}

# # Traffic light colors
# COLORS = {"RED": (0, 0, 255), "YELLOW": (0, 255, 255), "GREEN": (0, 255, 0)}

# # Create a resizable window
# cv2.namedWindow("Traffic Light Control System", cv2.WINDOW_NORMAL)

# # Vehicle classes in YOLO (2: car, 3: motorcycle, 5: bus, 7: truck)
# VEHICLE_CLASSES = [2, 3, 5, 7]

# def detect_vehicles(frame):
#     """Detect vehicles using YOLO and return count and frame with detections."""
#     results = model(frame)
#     vehicle_count = 0
#     vehicle_detections = frame.copy()

#     for result in results:
#         for box in result.boxes:
#             if box.cls in VEHICLE_CLASSES:
#                 vehicle_count += 1
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 cv2.rectangle(vehicle_detections, (x1, y1), (x2, y2), (0, 255, 0), 2)

#     return vehicle_count, vehicle_detections

# def update_traffic_lights(vehicle_counts):
#     """Update traffic light states based on vehicle density."""
#     max_lane = max(vehicle_counts, key=vehicle_counts.get)
#     max_vehicles = vehicle_counts[max_lane]
#     total_vehicles = sum(vehicle_counts.values())

#     green_time = max(5, min(20, int((max_vehicles / total_vehicles) * 20))) if total_vehicles else 10

#     for lane in traffic_lights:
#         if lane == max_lane:
#             if traffic_lights[lane] == "RED":
#                 yellow_timers[lane] = 2  # Yellow for 2 seconds before turning green
#                 traffic_lights[lane] = "YELLOW"
#             else:
#                 traffic_lights[lane] = "GREEN"
#                 green_timers[lane] = green_time
#         else:
#             if traffic_lights[lane] == "GREEN":
#                 yellow_timers[lane] = 2
#                 traffic_lights[lane] = "YELLOW"
#             else:
#                 traffic_lights[lane] = "RED"

# def draw_traffic_light(frame, lane, vehicle_count):
#     """Draw a thinner traffic light box with properly positioned lights."""
#     state = traffic_lights[lane]
#     x, y = 20, 20  # Traffic light position

#     # Thinner traffic light box
#     box_width = 30
#     box_height = 110

#     # Draw traffic light box
#     cv2.rectangle(frame, (x, y), (x + box_width, y + box_height), (50, 50, 50), -1)

#     # Draw traffic light pole
#     cv2.rectangle(frame, (x + 12, y + box_height), (x + 18, y + 200), (50, 50, 50), -1)

#     # Positions for RED, YELLOW, GREEN lights
#     circle_positions = [y + 20, y + 50, y + 80]

#     for i, color in enumerate(["RED", "YELLOW", "GREEN"]):
#         circle_color = COLORS[color] if state == color else (50, 50, 50)
#         cv2.circle(frame, (x + box_width // 2, circle_positions[i]), 10, circle_color, -1)

#     # Display lane information
#     cv2.putText(frame, lane, (x, y + box_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
#     cv2.putText(frame, f"Vehicles: {vehicle_count}", (x, y + box_height + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#     # Display signal time
#     time_left = green_timers[lane] if state == "GREEN" else (yellow_timers[lane] if state == "YELLOW" else "N/A")
#     cv2.putText(frame, f"{state}: {time_left}s", (x, y + box_height + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# try:
#     while True:
#         vehicle_counts = {"North": 0, "South": 0, "East": 0, "West": 0}
#         frames_to_display = []

#         for i, cap in enumerate(caps):
#             lane = list(traffic_lights.keys())[i]

#             if cap is None:
#                 frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
#             else:
#                 ret, new_frame = cap.read()
#                 if not ret:
#                     new_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
#                 else:
#                     new_frame = cv2.resize(new_frame, (FRAME_WIDTH, FRAME_HEIGHT))

#                 vehicle_counts[lane], processed_frame = detect_vehicles(new_frame)
#                 frame = processed_frame

#             draw_traffic_light(frame, lane, vehicle_counts[lane])
#             frames_to_display.append(frame)

#         # Handle traffic light transitions
#         for lane in yellow_timers:
#             if yellow_timers[lane] > 0:
#                 yellow_timers[lane] -= 1
#                 if yellow_timers[lane] == 0:
#                     if traffic_lights[lane] == "YELLOW":
#                         traffic_lights[lane] = "GREEN" if green_timers[lane] > 0 else "RED"

#         update_traffic_lights(vehicle_counts)

#         # Arrange frames in a 2x2 grid
#         top_row = np.hstack(frames_to_display[:2])
#         bottom_row = np.hstack(frames_to_display[2:])
#         combined_frame = np.vstack((top_row, bottom_row))

#         # Resize to match window size
#         win_width, win_height = cv2.getWindowImageRect("Traffic Light Control System")[2:]
#         combined_frame = cv2.resize(combined_frame, (win_width, win_height))

#         cv2.imshow("Traffic Light Control System", combined_frame)

#         # Decrease green timers
#         for lane in green_timers:
#             if green_timers[lane] > 0:
#                 green_timers[lane] -= 1

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# finally:
#     for cap in caps:
#         if cap:
#             cap.release()
#     cv2.destroyAllWindows()


import sys
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Get video paths from command-line arguments
video_paths = sys.argv[1:]  # Expecting 4 video paths

if len(video_paths) != 4:
    print("Error: Please provide exactly 4 video files as input.")
    sys.exit(1)

# Open video captures
caps = [cv2.VideoCapture(path) if path else None for path in video_paths]

# Frame size
FRAME_WIDTH, FRAME_HEIGHT = 640, 360

# Vehicle count tracking
vehicle_counts = [0] * len(caps)
vehicle_classifications = [{} for _ in range(len(caps))]

# COCO class names (Only relevant vehicle classes)
coco_classes = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# Create full-screen window
cv2.namedWindow("Traffic Management System", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Traffic Management System", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while any(cap.isOpened() for cap in caps):
    frames = []
    total_classifications = {c: 0 for c in coco_classes.values()}
    total_vehicle_count = 0

    for i, cap in enumerate(caps):
        if cap is None:
            frames.append(np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8))
            continue

        ret, frame = cap.read()
        if not ret:
            frames.append(np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8))
            continue

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        results = model(frame)

        vehicle_count = 0
        classifications = {}

        for obj in results[0].boxes.data:
            x1, y1, x2, y2, confidence, class_id = obj[:6]
            x1, y1, x2, y2, class_id = map(int, [x1, y1, x2, y2, class_id])
            if class_id in coco_classes and confidence > 0.5:
                vehicle_count += 1
                vehicle_type = coco_classes[class_id]
                classifications[vehicle_type] = classifications.get(vehicle_type, 0) + 1
                total_classifications[vehicle_type] += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{vehicle_type} {confidence:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        vehicle_counts[i] = vehicle_count
        vehicle_classifications[i] = classifications
        total_vehicle_count += vehicle_count

        cv2.putText(frame, f"Count: {vehicle_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        frames.append(frame)

    max_index = np.argmax(vehicle_counts)  # Find the lane with the highest traffic

    # Draw traffic signals
    for i, frame in enumerate(frames):
        red_color = (0, 0, 255) if i != max_index else (255, 255, 255)
        green_color = (0, 255, 0) if i == max_index else (255, 255, 255)

        cv2.circle(frame, (50, 80), 15, red_color, -1)  # Red light
        cv2.circle(frame, (50, 110), 15, green_color, -1)  # Green light

    # Summary panel
    summary_frame = np.zeros((70, FRAME_WIDTH * len(frames), 3), dtype=np.uint8)
    total_count_text = f"Total Vehicles: {total_vehicle_count}"
    class_text = " | ".join([f"{v}: {total_classifications.get(v, 0)}" for v in coco_classes.values()])
    cv2.putText(summary_frame, total_count_text, (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(summary_frame, class_text, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Arrange frames
    if len(frames) == 1:
        final_display = frames[0]
    elif len(frames) == 2:
        final_display = np.hstack(frames)
    else:
        mid = len(frames) // 2
        top_row = np.hstack(frames[:mid])
        bottom_row = np.hstack(frames[mid:]) if len(frames) > mid else np.zeros_like(top_row)
        final_display = np.vstack([top_row, bottom_row])

    summary_frame = cv2.resize(summary_frame, (final_display.shape[1], 70))
    final_display = np.vstack([summary_frame, final_display])

    cv2.imshow("Traffic Management System", final_display)

    # Press 'Q' or 'ESC' to exit
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
        break

# Release video captures
for cap in caps:
    if cap:
        cap.release()
cv2.destroyAllWindows()


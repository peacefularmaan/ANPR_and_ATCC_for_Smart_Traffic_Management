# import cv2
# import numpy as np
# from ultralytics import YOLO
# import time
# import datetime
# import os
# from collections import defaultdict
# import torch
# import pandas as pd
# from pathlib import Path
# import json

# class HighSpeedDetector:
#     def __init__(self):
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model = YOLO('yolov8n.pt')
#         self.model.to(self.device)
        
#         self.SPEED_LIMIT = 20 #SPEED LIMIT FOR VECHILES
#         self.DETECTION_LINES = [0.3, 0.7]
#         self.DISTANCE = 5
#         self.CONFIDENCE_THRESHOLD = 0.3
#         self.FRAME_SKIP = 1
#         self.VEHICLE_CLASSES = [2, 3, 5, 7]
        
#         self.vehicle_tracks = defaultdict(dict)
#         self.speed_records = []
        
#         self.setup_directories()
#         self.DEBUG = True

#     def setup_directories(self):
#         self.output_dir = Path("speed_monitoring")
#         self.violations_dir = self.output_dir / "violations"
#         self.data_dir = self.output_dir / "data"
        
#         for dir_path in [self.output_dir, self.violations_dir, self.data_dir]:
#             dir_path.mkdir(parents=True, exist_ok=True)
            
#         self.records_file = self.data_dir / "speed_records.csv"
#         if self.records_file.exists():
#             self.speed_records = pd.read_csv(self.records_file).to_dict('records')

#     def calculate_speed(self, time1, time2, y1, y2):
#         try:
#             time_diff = abs(time2 - time1)
#             if 0 < time_diff < 1.0:
#                 distance = abs(y2 - y1) * self.DISTANCE / 100
#                 speed = (distance / time_diff) * 3.6
                
#                 if 10 <= speed <= 150:
#                     return speed
#             return 0
#         except:
#             return 0

#     def save_violation(self, frame, vehicle_id, speed, box_coords):
#         timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
#         x1, y1, x2, y2 = map(int, box_coords)
#         vehicle_img = frame[max(0, y1-10):min(frame.shape[0], y2+10),
#                           max(0, x1-10):min(frame.shape[1], x2+10)]
        
#         img_path = self.violations_dir / f"violation_{timestamp}_{speed:.1f}kmh.jpg"
#         cv2.imwrite(str(img_path), vehicle_img)
        
#         record = {
#             'timestamp': timestamp,
#             'vehicle_id': vehicle_id,
#             'speed': speed,
#             'image_path': str(img_path)
#         }
        
#         self.speed_records.append(record)
#         pd.DataFrame(self.speed_records).to_csv(self.records_file, index=False)
        
#         if self.DEBUG:
#             print(f"Violation recorded: {speed:.1f} km/h")
        
#         return record

#     def process_frame(self, frame, frame_number, fps):
#         height, width = frame.shape[:2]
#         frame = cv2.resize(frame, (640, 480))
#         height, width = frame.shape[:2]
        
#         lines = [int(height * pos) for pos in self.DETECTION_LINES]
        
#         for line in lines:
#             cv2.line(frame, (0, line), (width, line), (255, 0, 0), 2)
        
#         results = self.model.track(
#             frame,
#             persist=True,
#             conf=self.CONFIDENCE_THRESHOLD,
#             classes=self.VEHICLE_CLASSES,
#             verbose=False
#         )
        
#         if not results or not results[0].boxes:
#             return frame

#         current_time = time.time()
        
#         for box in results[0].boxes:
#             if not box.id:
#                 continue
                
#             vehicle_id = int(box.id)
#             box_data = box.xyxy[0].cpu().numpy()
#             confidence = float(box.conf)
            
#             if confidence < self.CONFIDENCE_THRESHOLD:
#                 continue

#             x1, y1, x2, y2 = map(int, box_data[:4])
#             center_y = (y1 + y2) / 2

#             if vehicle_id not in self.vehicle_tracks:
#                 self.vehicle_tracks[vehicle_id] = {
#                     'first_detection': {'time': current_time, 'position': center_y}
#                 }
#             else:
#                 first_detection = self.vehicle_tracks[vehicle_id]['first_detection']
#                 time_diff = current_time - first_detection['time']
                
#                 if time_diff > 0.1:
#                     speed = self.calculate_speed(
#                         first_detection['time'],
#                         current_time,
#                         first_detection['position'],
#                         center_y
#                     )
                    
#                     if speed > self.SPEED_LIMIT:
#                         self.save_violation(frame, vehicle_id, speed, (x1, y1, x2, y2))
#                         color = (0, 0, 255)
#                     else:
#                         color = (0, 255, 0)
                    
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                     cv2.putText(frame, f"{speed:.1f} km/h", (x1, y1-10),
#                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
#                     self.vehicle_tracks[vehicle_id] = {
#                         'first_detection': {'time': current_time, 'position': center_y}
#                     }

#         return frame

#     def process_video(self, video_path):
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             print("Error: Could not open video.")
#             return

#         frame_count = 0
#         start_time = time.time()
        
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             frame_count += 1
            
#             if frame_count % self.FRAME_SKIP != 0:
#                 continue

#             processed_frame = self.process_frame(frame, frame_count, cap.get(cv2.CAP_PROP_FPS))
            
#             cv2.imshow("High Speed Detection", processed_frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         total_time = time.time() - start_time
#         self.save_statistics(frame_count, total_time)
        
#         cap.release()
#         cv2.destroyAllWindows()

#     def save_statistics(self, frame_count, total_time):
#         stats = {
#             'total_frames': frame_count,
#             'total_time': total_time,
#             'average_fps': frame_count / total_time,
#             'violations_detected': len(self.speed_records),
#             'highest_speed': max([record['speed'] for record in self.speed_records]) if self.speed_records else 0
#         }
        
#         with open(self.data_dir / 'statistics.json', 'w') as f:
#             json.dump(stats, f, indent=4)
        
#         if self.DEBUG:
#             print("\nProcessing Statistics:")
#             for key, value in stats.items():
#                 print(f"{key}: {value}")

# def main():
#     detector = HighSpeedDetector()
#     video_path = input("Enter video path: ")
#     detector.process_video(video_path)

# if __name__ == "__main__":
#     main()


import sys
import cv2
import numpy as np
import time
import datetime
import os
import torch
import json
from collections import defaultdict
from ultralytics import YOLO
from pathlib import Path

# Suppress terminal on Windows (for pythonw.exe)
if os.name == "nt":
    import ctypes
    ctypes.windll.kernel32.FreeConsole()

# Get video file path from arguments or use default
video_path = sys.argv[1] if len(sys.argv) > 1 else "speed_test.mp4"

# Open the video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Unable to open video file {video_path}")
    sys.exit(1)

# Load YOLO model
model = YOLO("yolov8n.pt")

# Speed limit in km/h
SPEED_LIMIT = 20
DISTANCE = 5  # Distance covered between detection lines (in meters)
CONFIDENCE_THRESHOLD = 0.3
FRAME_SKIP = 1

# Detection line positions (percentage of frame height)
DETECTION_LINES = [0.3, 0.7]
VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO: Car, Motorcycle, Bus, Truck

# Dictionary to track vehicle speed calculations
vehicle_tracks = defaultdict(dict)

# OpenCV window setup
cv2.namedWindow("Overspeeding Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Overspeeding Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def calculate_speed(time1, time2, y1, y2):
    try:
        time_diff = abs(time2 - time1)
        if 0 < time_diff < 1.0:
            distance = abs(y2 - y1) * DISTANCE / 100
            speed = (distance / time_diff) * 3.6  # Convert m/s to km/h
            return speed if 10 <= speed <= 150 else 0
    except:
        pass
    return 0

while True:
    success, frame = cap.read()
    if not success:
        break

    height, width = frame.shape[:2]
    frame = cv2.resize(frame, (640, 480))
    height, width = frame.shape[:2]

    # Draw detection lines
    lines = [int(height * pos) for pos in DETECTION_LINES]
    for line in lines:
        cv2.line(frame, (0, line), (width, line), (255, 0, 0), 2)

    # Run YOLO model on frame
    results = model.track(
        frame, persist=True, conf=CONFIDENCE_THRESHOLD, classes=VEHICLE_CLASSES, verbose=False
    )

    if not results or not results[0].boxes:
        cv2.imshow("Overspeeding Detection", frame)
        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:  # Quit on 'q' or ESC
            break
        continue

    current_time = time.time()

    for box in results[0].boxes:
        if not box.id:
            continue

        vehicle_id = int(box.id)
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        center_y = (y1 + y2) / 2

        # First detection of the vehicle
        if vehicle_id not in vehicle_tracks:
            vehicle_tracks[vehicle_id] = {'time': current_time, 'position': center_y}
        else:
            first_detection = vehicle_tracks[vehicle_id]
            speed = calculate_speed(
                first_detection['time'], current_time, first_detection['position'], center_y
            )

            if speed > SPEED_LIMIT:
                color = (0, 0, 255)  # Red for overspeeding
            else:
                color = (0, 255, 0)  # Green for normal speed

            # Draw bounding box and speed text
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{speed:.1f} km/h", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Update the vehicle's tracking data
            vehicle_tracks[vehicle_id] = {'time': current_time, 'position': center_y}

    # Show output frame
    cv2.imshow("Overspeeding Detection", frame)

    # Quit on 'q' or ESC
    if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

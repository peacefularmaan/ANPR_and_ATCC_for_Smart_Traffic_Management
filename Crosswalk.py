# import cv2
# import numpy as np
# from ultralytics import YOLO

# video_path = r"C:\Users\babur\Downloads\crosswalk1.mp4"  # Video path

# def load_model(model_path="yolov8n.pt"):
#     return YOLO(model_path)

# def detect_pedestrians(frame, model):
#     results = model(frame)
#     pedestrian_detected = False
    
#     for obj in results[0].boxes.data:
#         class_id = int(obj[5])
#         confidence = obj[4]
#         x1, y1, x2, y2 = map(int, obj[:4])

#         if class_id == 0 and confidence > 0.5:  # Class ID 0 is for pedestrians
#             pedestrian_detected = True
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, f"Pedestrian {confidence:.2f}", (x1, y1 - 5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#     return frame, pedestrian_detected

# def draw_traffic_light(frame, traffic_light):
#     light_x, light_y = 50, 50
#     light_width, light_height = 100, 250
#     red_light_center = (light_x + light_width // 2, light_y + 70)
#     green_light_center = (light_x + light_width // 2, light_y + 180)
#     circle_radius = 40
    
#     cv2.rectangle(frame, (light_x, light_y), (light_x + light_width, light_y + light_height), (50, 50, 50), -1)
    
#     if traffic_light == "Red":
#         cv2.circle(frame, red_light_center, circle_radius, (0, 0, 255), -1)
#         cv2.circle(frame, green_light_center, circle_radius, (255, 255, 255), -1)
#     else:
#         cv2.circle(frame, red_light_center, circle_radius, (255, 255, 255), -1)
#         cv2.circle(frame, green_light_center, circle_radius, (0, 255, 0), -1)
    
#     return frame

# def process_video(video_path, model):
#     cap = cv2.VideoCapture(video_path)
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         frame, pedestrian_detected = detect_pedestrians(frame, model)
#         traffic_light = "Red" if pedestrian_detected else "Green"
#         frame = draw_traffic_light(frame, traffic_light)
        
#         frame = cv2.resize(frame, (900, 700))
#         cv2.imshow("Smart Crosswalk System", frame)
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     model = load_model()
#     process_video(video_path, model)


# import sys
# import cv2
# import numpy as np
# from ultralytics import YOLO
# import os

# # ✅ Suppress terminal on Windows
# if os.name == "nt":
#     import ctypes
#     ctypes.windll.kernel32.FreeConsole()

# # ✅ Load YOLOv8 Model
# def load_model(model_path="yolov8n.pt"):
#     return YOLO(model_path)

# # ✅ Get video path from CLI or use default
# video_path = sys.argv[1] if len(sys.argv) > 1 else "crosswalk1.mp4"

# def detect_pedestrians(frame, model):
#     """Detect pedestrians in the frame using YOLOv8."""
#     results = model(frame)
#     pedestrian_detected = False

#     for obj in results[0].boxes.data:
#         class_id = int(obj[5])
#         confidence = obj[4]
#         x1, y1, x2, y2 = map(int, obj[:4])

#         if class_id == 0 and confidence > 0.5:  # Class ID 0 is for pedestrians
#             pedestrian_detected = True
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, f"Pedestrian {confidence:.2f}", (x1, y1 - 5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#     return frame, pedestrian_detected

# def draw_traffic_light(frame, pedestrian_detected):
#     """Draws a traffic light with red or green indication based on pedestrian detection."""
#     light_x, light_y = 50, 50
#     light_width, light_height = 100, 250
#     red_light_center = (light_x + light_width // 2, light_y + 70)
#     green_light_center = (light_x + light_width // 2, light_y + 180)
#     circle_radius = 40

#     cv2.rectangle(frame, (light_x, light_y), (light_x + light_width, light_y + light_height), (50, 50, 50), -1)

#     if pedestrian_detected:
#         cv2.circle(frame, red_light_center, circle_radius, (0, 0, 255), -1)  # Red ON
#         cv2.circle(frame, green_light_center, circle_radius, (255, 255, 255), -1)  # Green OFF
#     else:
#         cv2.circle(frame, red_light_center, circle_radius, (255, 255, 255), -1)  # Red OFF
#         cv2.circle(frame, green_light_center, circle_radius, (0, 255, 0), -1)  # Green ON

#     return frame

# def process_video(video_path, model):
#     """Process video frame-by-frame and display pedestrian detection with traffic lights."""
#     cap = cv2.VideoCapture(video_path)

#     if not cap.isOpened():
#         print(f"❌ Error: Unable to open video file {video_path}")
#         return

#     # ✅ Open full-screen OpenCV window
#     cv2.namedWindow("Smart Crosswalk System", cv2.WINDOW_NORMAL)
#     cv2.setWindowProperty("Smart Crosswalk System", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame, pedestrian_detected = detect_pedestrians(frame, model)
#         frame = draw_traffic_light(frame, pedestrian_detected)

#         frame = cv2.resize(frame, (900, 700))
#         cv2.imshow("Smart Crosswalk System", frame)

#         # ✅ Press 'q' or ESC to exit
#         key = cv2.waitKey(1)
#         if key == ord('q') or key == 27:  # 27 is ESC key
#             break

#     # ✅ **Hold video at end until manual close**
#     while True:
#         key = cv2.waitKey(0)
#         if key == ord('q') or key == 27:
#             break

#     print("✅ Video finished. Press any key to exit.")
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     model = load_model()
#     process_video(video_path)


import sys
import cv2
import numpy as np
from ultralytics import YOLO
import os

# Suppress console window on Windows
if os.name == "nt":
    import ctypes
    ctypes.windll.kernel32.FreeConsole()

# Get video path from command-line arguments or use default
video_path = sys.argv[1] if len(sys.argv) > 1 else "crosswalk1.mp4"

def load_model(model_path="yolov8n.pt"):
    """Load YOLOv8 model for pedestrian detection."""
    return YOLO(model_path)

def detect_pedestrians(frame, model):
    """Detects pedestrians in the frame using YOLO."""
    results = model(frame)
    pedestrian_detected = False
    
    for obj in results[0].boxes.data:
        class_id = int(obj[5])
        confidence = obj[4]
        x1, y1, x2, y2 = map(int, obj[:4])

        if class_id == 0 and confidence > 0.5:  # Class ID 0 is for pedestrians
            pedestrian_detected = True
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Pedestrian {confidence:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame, pedestrian_detected

def draw_traffic_light(frame, traffic_light):
    """Draws traffic light indicator on the frame."""
    light_x, light_y = 50, 50
    light_width, light_height = 100, 250
    red_light_center = (light_x + light_width // 2, light_y + 70)
    green_light_center = (light_x + light_width // 2, light_y + 180)
    circle_radius = 40

    # Draw the light box
    cv2.rectangle(frame, (light_x, light_y), (light_x + light_width, light_y + light_height), (50, 50, 50), -1)

    # Red light ON if pedestrian detected, else Green light ON
    if traffic_light == "Red":
        cv2.circle(frame, red_light_center, circle_radius, (0, 0, 255), -1)  # Red ON
        cv2.circle(frame, green_light_center, circle_radius, (255, 255, 255), -1)  # Green OFF
    else:
        cv2.circle(frame, red_light_center, circle_radius, (255, 255, 255), -1)  # Red OFF
        cv2.circle(frame, green_light_center, circle_radius, (0, 255, 0), -1)  # Green ON
    
    return frame

def process_video(video_path, model):
    """Processes the input video, detects pedestrians, and controls traffic lights."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    # Open video in fullscreen mode
    cv2.namedWindow("Smart Crosswalk System", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Smart Crosswalk System", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        frame, pedestrian_detected = detect_pedestrians(frame, model)
        traffic_light = "Red" if pedestrian_detected else "Green"
        frame = draw_traffic_light(frame, traffic_light)

        frame = cv2.resize(frame, (1280, 720))  # Full HD resolution
        cv2.imshow("Smart Crosswalk System", frame)

        # Press 'q' or ESC to exit
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            break

    print("Video processing completed. Exiting...")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model = load_model()
    process_video(video_path, model)


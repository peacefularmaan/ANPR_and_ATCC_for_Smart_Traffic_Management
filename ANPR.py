# import sys
# import cv2
# import logging
# import pytesseract
# from ultralytics import YOLO
# import os
# import csv

# # Suppress terminal on Windows
# if os.name == "nt":
#     import ctypes
#     ctypes.windll.kernel32.FreeConsole()

# # Configure logging
# logging.getLogger('ultralytics').setLevel(logging.WARNING)  # Suppress YOLO logs

# # Set Tesseract OCR path (ensure Tesseract is installed)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# # Load YOLO models
# vehicle_model = YOLO("yolov8n.pt")  # YOLOv8 for vehicle detection (pre-trained)
# plate_model = YOLO("best.pt")        # Fine-tuned YOLO model for license plate detection

# # COCO dataset vehicle class IDs: Car=2, Motorcycle=3, Bus=5, Truck=7
# VEHICLE_CLASSES = [2, 3, 5, 7]

# no_plates = []
# frame_data = []

# # Get video path from command-line arguments or use default
# video_path = sys.argv[1] if len(sys.argv) > 1 else "test_video.mp4"

# def detect_vehicles_and_plates(video_path, frame_skip=8):
#     cap = cv2.VideoCapture(video_path)
    
#     if not cap.isOpened():
#         print(f"Error: Unable to open video file {video_path}")
#         return

#     output_dir = "processed_output"
#     os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

#     frame_count = 0

#     # Create OpenCV display window
#     cv2.namedWindow("ANPR - Vehicle & Plate Detection", cv2.WINDOW_NORMAL)
#     cv2.setWindowProperty("ANPR - Vehicle & Plate Detection", cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break  # Exit loop when the video ends

#         frame_count += 1
#         if frame_count % frame_skip != 0:
#             continue  # Skip frames for performance

#         # Detect vehicles in the frame
#         vehicle_results = vehicle_model(frame)

#         for result in vehicle_results:
#             for i, box in enumerate(result.boxes.xyxy):
#                 x1, y1, x2, y2 = map(int, box)
#                 cls = int(result.boxes.cls[i].item())  # Class ID
#                 confidence = result.boxes.conf[i].item()  # Confidence score

#                 if cls in VEHICLE_CLASSES and confidence > 0.3:
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box for vehicles
#                     cv2.putText(frame, f"Vehicle {confidence:.2f}", (x1, y1 - 5),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#                     # Crop vehicle region for license plate detection
#                     vehicle_roi = frame[y1:y2, x1:x2]

#                     # Detect license plates within the detected vehicle
#                     plate_results = plate_model(vehicle_roi)

#                     for plate_result in plate_results:
#                         for j, plate_box in enumerate(plate_result.boxes.xyxy):
#                             px1, py1, px2, py2 = map(int, plate_box)
#                             p_conf = plate_result.boxes.conf[j].item()

#                             if p_conf > 0.3:
#                                 # Draw bounding box for the plate (Green)
#                                 cv2.rectangle(vehicle_roi, (px1, py1), (px2, py2), (0, 255, 0), 2)
#                                 cv2.putText(vehicle_roi, f"Plate {p_conf:.2f}", (px1, py1 - 5),
#                                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                
#                                 plate_roi = vehicle_roi[py1:py2, px1:px2]

#                                 # OCR for plate number extraction
#                                 plate_text = pytesseract.image_to_string(plate_roi, config='--psm 7')
#                                 lcs_plt = plate_text.strip()

#                                 # Save extracted plates
#                                 if len(lcs_plt) > 6 and lcs_plt not in no_plates:
#                                     no_plates.append(lcs_plt)
#                                     frame_data.append(frame_count)
#                                     image_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
#                                     cv2.imwrite(image_path, vehicle_roi)

#         # Display processed frame
#         cv2.imshow("ANPR - Vehicle & Plate Detection", frame)

#         # Press 'q' or ESC to exit
#         key = cv2.waitKey(1)
#         if key == ord('q') or key == 27:  # 27 is ESC key
#             break

#     # ✅ **NEW FIX: Wait at the end if video finishes automatically**
#     while True:
#         key = cv2.waitKey(0)
#         if key == ord('q') or key == 27:
#             break

#     print("Video finished. Press any key to exit.")
#     cap.release()
#     cv2.destroyAllWindows()

#     # Save extracted plate data to CSV
#     save_plate_data()

# def save_plate_data():
#     csv_path = "processed_output/speed_records.csv"
#     field_names = ['No.', 'License-plate', 'Frame-no.']
#     data_csv = [{'No.': i, 'License-plate': no_plates[i], 'Frame-no.': frame_data[i]} for i in range(len(no_plates))]

#     with open(csv_path, 'w', newline='') as csvfile:
#         writer = csv.DictWriter(csvfile, fieldnames=field_names)
#         writer.writeheader()
#         writer.writerows(data_csv)

#     print(f"License plate data saved to {csv_path}")

# if __name__ == "__main__":
#     detect_vehicles_and_plates(video_path)






import sys
import cv2
import logging
import pytesseract
from ultralytics import YOLO
import os
import csv

# 🛑 Suppress terminal on Windows
if os.name == "nt":
    import ctypes
    ctypes.windll.kernel32.FreeConsole()

# ✅ Configure logging
logging.getLogger('ultralytics').setLevel(logging.WARNING)  # Suppress YOLO logs

# ✅ Set Tesseract OCR path (ensure Tesseract is installed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ✅ Load YOLO models
vehicle_model = YOLO("yolov8n.pt")  # YOLOv8 for vehicle detection (pre-trained)
plate_model = YOLO("best.pt")        # Fine-tuned YOLO model for license plate detection

# ✅ COCO dataset vehicle class IDs: Car=2, Motorcycle=3, Bus=5, Truck=7
VEHICLE_CLASSES = [2, 3, 5, 7]

no_plates = []
frame_data = []

# ✅ Get video path from CLI or use default
video_path = sys.argv[1] if len(sys.argv) > 1 else "test_video.mp4"

def detect_vehicles_and_plates(video_path, frame_skip=8):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Error: Unable to open video file {video_path}")
        return

    output_dir = "processed_output"
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    frame_count = 0

    # ✅ Open full-screen OpenCV window
    cv2.namedWindow("ANPR - Vehicle & Plate Detection", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("ANPR - Vehicle & Plate Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop when the video ends

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # Skip frames for performance

        # ✅ Detect vehicles in the frame
        vehicle_results = vehicle_model(frame)

        for result in vehicle_results:
            for i, box in enumerate(result.boxes.xyxy):
                x1, y1, x2, y2 = map(int, box)
                cls = int(result.boxes.cls[i].item())  # Class ID
                confidence = result.boxes.conf[i].item()  # Confidence score

                if cls in VEHICLE_CLASSES and confidence > 0.3:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box for vehicles
                    cv2.putText(frame, f"Vehicle {confidence:.2f}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    # ✅ Crop vehicle region for license plate detection
                    vehicle_roi = frame[y1:y2, x1:x2]

                    # ✅ Detect license plates within the detected vehicle
                    plate_results = plate_model(vehicle_roi)

                    for plate_result in plate_results:
                        for j, plate_box in enumerate(plate_result.boxes.xyxy):
                            px1, py1, px2, py2 = map(int, plate_box)
                            p_conf = plate_result.boxes.conf[j].item()

                            if p_conf > 0.3:
                                # ✅ Draw bounding box for the plate (Green)
                                cv2.rectangle(vehicle_roi, (px1, py1), (px2, py2), (0, 255, 0), 2)
                                cv2.putText(vehicle_roi, f"Plate {p_conf:.2f}", (px1, py1 - 5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                
                                plate_roi = vehicle_roi[py1:py2, px1:px2]

                                # ✅ OCR for plate number extraction
                                plate_text = pytesseract.image_to_string(plate_roi, config='--psm 7')
                                lcs_plt = plate_text.strip()

                                # ✅ Save extracted plates
                                if len(lcs_plt) > 6 and lcs_plt not in no_plates:
                                    no_plates.append(lcs_plt)
                                    frame_data.append(frame_count)
                                    image_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
                                    cv2.imwrite(image_path, vehicle_roi)

        # ✅ Display processed frame in full screen
        cv2.imshow("ANPR - Vehicle & Plate Detection", frame)

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

    # ✅ Save extracted plate data to CSV
    save_plate_data()

def save_plate_data():
    csv_path = "processed_output/plate_records.csv"
    field_names = ['No.', 'License-plate', 'Frame-no.']
    data_csv = [{'No.': i, 'License-plate': no_plates[i], 'Frame-no.': frame_data[i]} for i in range(len(no_plates))]

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(data_csv)

    print(f"✅ License plate data saved to {csv_path}")

if __name__ == "__main__":
    detect_vehicles_and_plates(video_path)

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import subprocess

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Feature-based detections (For Features Page)
FEATURE_SCRIPTS = {
    'Helmet Detection': 'Helmet_Detector.py',
    'Triple Riding Detection': 'TripleRiding.py',
    'Accident Detection': 'Accident_Detector.py',
    'Overspeeding Detection': 'Overspeeding.py',
    'Heatmap Visualization': 'Heatmap_Visualizer.py',
    'Emergency Vehicle Detection': 'Emergency_Vehicle_Detector.py',
    'Crosswalk Detection': 'Crosswalk.py',
}

# Process-based detections (For Live Monitoring)
PROCESS_SCRIPTS = {
    'ANPR': 'ANPR.py',
    'ATCC': 'ATCC.py',
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return jsonify({'message': 'Flask backend is running!'})

@app.route('/process', methods=['POST'])
def process_video():
    process = request.form.get('process')  # For Live Monitoring (ANPR/ATCC)
    feature = request.form.get('feature')  # For Features Page

    # Ensure either process or feature is selected
    if not process and not feature:
        return jsonify({'error': 'No process or feature selected'}), 400

    # 📌 **Handling ATCC (Multiple Videos Allowed)**
    if process == "ATCC":
        files = request.files.getlist('videos')

        if not files or len(files) == 0:
            return jsonify({'error': 'Please upload at least one video file for ATCC'}), 400

        video_paths = []
        for file in files:
            if not allowed_file(file.filename):
                return jsonify({'error': f'Invalid file: {file.filename}. Allowed: MP4, AVI, MOV, MKV'}), 400

            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            video_paths.append(file_path)

        print(f"✅ ATCC Videos Received: {video_paths}")

        try:
            subprocess.Popen(['python', PROCESS_SCRIPTS[process]] + video_paths, creationflags=subprocess.CREATE_NEW_CONSOLE)
            return jsonify({'message': f'{process} started successfully!', 'videos': video_paths})
        except Exception as e:
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500

    # 📌 **Handling ANPR (Only 1 Video)**
    elif process == "ANPR":
        file = request.files.get('videos')

        if not file or not allowed_file(file.filename):
            return jsonify({'error': 'Please upload exactly one video file for ANPR'}), 400

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        print(f"✅ ANPR Video Received: {file_path}")

        try:
            subprocess.Popen(['python', PROCESS_SCRIPTS[process], file_path], creationflags=subprocess.CREATE_NEW_CONSOLE)
            return jsonify({'message': f'{process} started successfully!', 'video': file.filename})
        except Exception as e:
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500

    # 📌 **Handling Features Page (Single Video Only)**
    elif feature:
        file = request.files.get('video')  # Single video input

        if not file or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid or missing file. Allowed: MP4, AVI, MOV, MKV'}), 400

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        script_to_run = FEATURE_SCRIPTS.get(feature)
        if not script_to_run:
            return jsonify({'error': 'Invalid feature selection'}), 400

        print(f"✅ Feature '{feature}' Video Received: {file_path}")

        try:
            subprocess.Popen(['python', script_to_run, file_path], creationflags=subprocess.CREATE_NEW_CONSOLE)
            return jsonify({'message': f'{feature} started successfully!', 'video': file.filename})
        except Exception as e:
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500

    return jsonify({'error': 'Invalid request'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

import sys
import os
# Add VSVIG to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'VSVIG'))
from flask import Flask, render_template, request, send_from_directory, jsonify
import cv2
import random
from VSVIG.testt import seizure_detection
from fall_detection import fall_detection
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='static')
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    # Check if the video file is in the request
    if 'video' not in request.files:
        return jsonify(success=False, error="No file part")

    file = request.files['video']

    # If no file is selected, return an error
    if file.filename == '':
        return jsonify(success=False, error="No selected file")

    # Secure the filename to prevent directory traversal attacks
    filename = secure_filename(file.filename)
    
    # Save the file to the upload folder
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return jsonify(success=True, filename=filename)
    except Exception as e:
        return jsonify(success=False, error=f"Error saving file: {str(e)}")

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out.write(cv2.merge([gray, gray, gray]))
    
    cap.release()
    out.release()

@app.route('/trigger_log')
def trigger_log():
    # Simulate detection: generate a random probability between 0 and 1 for each second
    probabilities = [random.uniform(0, 1) for _ in range(10)]  # Simulating 10 seconds of data
    logs = []

    # For each second, determine if it's a "Fall Detected" or "No fall"
    for i, prob in enumerate(probabilities):
        if prob >= 0.8:  # Threshold for fall detection
            logs.append(f"Fall Detected (Probability: {prob:.2f})")

    # Return all logs as a JSON response
    return jsonify(logs=logs)

@app.route('/trigger_seizure_log', methods=['POST'])
def trigger_seizure_log():
    data = request.json
    filename = data.get('filename')
    # Simulate seizure detection with random probabilities (from 0 to 1)
    probabilities = seizure_detection()  # Simulating 10 seconds of data
    logs = []

    # # For each second, check if it's a seizure (probability > 0.7) or not
    for i, prob in enumerate(probabilities):
        if prob < 0.5:
            prob = prob + 0.8
            if prob > 1:
                prob = 0.99
        else:
            prob = prob
        logs.append(float(prob))

    # Return the seizure logs as a JSON response
    # temp = probabilities.tolist()
    return jsonify(success=True, probabilities=logs)

@app.route('/trigger_fall_log', methods=['POST'])
def trigger_fall_log():
    data = request.json
    filename = data.get('filename')
    # Simulate seizure detection with random probabilities (from 0 to 1)
    detection = fall_detection()  # Simulating 10 seconds of data

    return jsonify(success=True, detection=detection)

if __name__ == '__main__':
    app.run(debug=True)

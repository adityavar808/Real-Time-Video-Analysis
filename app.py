from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import os
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
import logging
from werkzeug.contrib.fixers import ProxyFix
# from deepface import DeepFace 

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Configure caching
cache = Cache(app, config={
    'CACHE_TYPE': 'simple',
    'CACHE_DEFAULT_TIMEOUT': 300
})

# Global camera object
camera = None

def get_camera():
    global camera
    if camera is None:
        try:
            camera = cv2.VideoCapture(0)
        except Exception as e:
            logger.error(f"Error initializing camera: {str(e)}")
            return None
    return camera

def release_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None

@app.before_request
def before_request():
    if request.headers.get('X-Forwarded-Proto') == 'http':
        url = request.url.replace('http://', 'https://', 1)
        return redirect(url, code=301)

user_logged_in = True  # Toggle login state

def generate_frames():
    try:
        camera = get_camera()
        if camera is None:
            return

        while True:
            success, frame = camera.read()
            if not success:
                logger.error("Failed to read frame from camera")
                break
            else:
                try:
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                except Exception as e:
                    logger.error(f"Error processing frame: {str(e)}")
                    break
    except Exception as e:
        logger.error(f"Error in generate_frames: {str(e)}")
    finally:
        release_camera()

@app.route('/')
def home():
    return render_template('home.html', user_logged_in=user_logged_in)

@app.route('/object')
def object():
    return render_template('object.html', user_logged_in=user_logged_in)

@app.route('/object/video_feed')
def object_video_feed():
    def generate_object_frames():
        net = cv2.dnn.readNet("python_Scripts\\yolov3.weights", "python_Scripts\\yolov3.cfg")  # Path to YOLO weights and config file
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

        # Load class labels (coco.names should be in the same folder)
        with open("python_Scripts\\coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]

        # Open the webcam (0 is the default camera)
        cap = cv2.VideoCapture(0)

        # Check if the webcam was opened correctly
        if not cap.isOpened():
            print("Error: Could not access the webcam.")
            exit()

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read frame from webcam.")
                break
            
            # Prepare the frame for YOLO
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            # Initialize lists for detection results
            boxes = []
            confidences = []
            class_ids = []

            height, width, channels = frame.shape

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    # Set a confidence threshold (0.5)
                    if confidence > 0.5:  
                        # Get the bounding box
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # Apply non-maxima suppression to eliminate redundant overlapping boxes
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            # Draw rectangles around detected objects and label them
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    label = classes[class_ids[i]]  # Get the class name from the label
                    confidence = confidences[i]

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle
                    cv2.putText(frame, f"{label} ({round(confidence * 100, 2)}%)", (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Encode frame to bytes
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
        # When everything is done, release the capture and close windows
        cap.release()

    return Response(generate_object_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/emotion')
def emotion():
    return render_template('Emotion.html', user_logged_in=user_logged_in)


@app.route('/emotion/video_feed')
def emotion_video_feed():
    def generate_emotion_frame():
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        # Open webcam
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

            for (x, y, w, h) in faces:
                # Draw a rectangle around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Extract the region of interest (ROI)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]

                # Detect eyes within the face ROI
                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

                # Detect smile using thresholding and contours
                mouth_region = roi_gray[int(h / 2):h, :]
                _, mouth_thresh = cv2.threshold(mouth_region, 70, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(mouth_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                # Analyze contours to detect smiles
                smiling = any(cv2.contourArea(c) > 300 for c in contours)

                # Emotion inference based on eye and smile detection
                if len(eyes) >= 2 and smiling:
                    emotion = "Happy"
                elif len(eyes) < 2 and not smiling:
                    emotion = "Sad"
                else:
                    emotion = "Neutral"

                # Display the detected emotion
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Encode the frame for streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        cap.release()

    return Response(generate_emotion_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/human')
def human():
    return render_template('human.html', user_logged_in=user_logged_in)

@app.route('/human/video_feed')
def human_video_feed():
    def generate_human_frames():
        # Load pre-trained YOLO model (weights and config)
        net = cv2.dnn.readNet("python_Scripts/yolov3.weights", "python_Scripts/yolov3.cfg")
        with open("python_Scripts/coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]

        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not access the webcam.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Prepare frame for YOLO
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            # Analyze detections
            human_count = 0
            height, width, _ = frame.shape
            boxes = []
            confidences = []
            class_ids = []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > 0.5 and class_id == 0:  # 0 corresponds to 'person' class
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # Draw rectangles around detected humans
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    human_count += 1

            # Display the human count
            cv2.putText(frame, f'Humans detected: {human_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Encode frame to bytes
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        cap.release()

    return Response(generate_human_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/sign')
def sign():
    return render_template('sign.html', user_logged_in=user_logged_in)

@app.route('/sign/video_feed')
def sign_video_feed():
    def generate_sign_frames():
        pass
    return Response(generate_sign_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/vehicle')
def vehicle():
    return render_template('vehicle.html', user_logged_in=user_logged_in)

@app.route('/vehicle/video_feed')
def vehicle_video_feed():
    def generate_vehicle_frames():
        net = cv2.dnn.readNet("python_Scripts/yolov3.weights", "python_Scripts/yolov3.cfg")  
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

        with open("python_Scripts/coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]

        # Open the webcam (0 is the default camera)
        cap = cv2.VideoCapture(0)

        # Check if the webcam was opened correctly
        if not cap.isOpened():
            print("Error: Could not access the webcam.")
            exit()

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read frame from webcam.")
                break
            
            # Prepare the frame for YOLO
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            # Analyze the detections
            vehicle_count = 0
            height, width, channels = frame.shape
            boxes = []
            confidences = []
            class_ids = []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > 0.5 and (class_id in (2,3,5,7)):  # 0 corresponds to 'car','motorbike','bus','truck' classes in COCO dataset
                        # Get the bounding box
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # Apply non-maxima suppression to eliminate redundant overlapping boxes
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            # Draw rectangles around detected humans
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    vehicle_count += 1

            # Display the human count
            cv2.putText(frame, f'Vehicles detected: {vehicle_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Encode frame to bytes
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
        # When everything is done, release the capture and close windows
        cap.release()

    return Response(generate_vehicle_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/movement')
def movement():
    return render_template('movement.html', user_logged_in=user_logged_in)

@app.route('/movement/video_feed')
def movement_video_feed():
    def generate_movement_frames():
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not access the webcam.")
            return

        # Initialize the MOG2 background subtractor
        mog_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame from the webcam.")
                    break

                # Resize the frame for faster processing
                frame_resized = cv2.resize(frame, (640, 480))

                # Apply Gaussian Blur to reduce noise
                blurred_frame = cv2.GaussianBlur(frame_resized, (5, 5), 0)

                # Apply the MOG2 background subtraction
                learning_rate = 0.01
                foreground_mask = mog_subtractor.apply(blurred_frame, learningRate=learning_rate)

                # Apply binary thresholding to refine the mask
                _, thresholded_mask = cv2.threshold(foreground_mask, 200, 255, cv2.THRESH_BINARY)

                # Morphological operations to further clean up the mask
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                clean_mask = cv2.morphologyEx(thresholded_mask, cv2.MORPH_CLOSE, kernel)

                # Find contours for object detection
                contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour) > 1000:  # Filter by area
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(frame_resized, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Encode frame to bytes
                ret, buffer = cv2.imencode('.jpg', clean_mask)
                if not ret:
                    print("Error: Could not encode frame.")
                    break

                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        finally:
            # Ensure the webcam is released even if an exception occurs
            cap.release()

    return Response(generate_movement_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.errorhandler(429)
def ratelimit_handler(e):
    return {"error": "rate limit exceeded"}, 429

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return {"error": "internal server error"}, 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

import cv2
import firebase_admin
from firebase_admin import credentials, storage, db
from ultralytics import YOLO
import numpy as np
import uuid
import threading
from datetime import datetime

# Function to handle uploading frames in a separate thread
def upload_frame_to_firebase(frame_bytes, estimated_size, detection_time):
    # Create a unique filename
    filename = f"frames/frame_{uuid.uuid4()}.jpg"

    # Upload to Firebase Storage
    blob = bucket.blob(filename)
    blob.upload_from_string(frame_bytes, content_type='image/jpeg')

    # Get the URL of the uploaded frame
    frame_url = blob.public_url

    # Save metadata (e.g., pothole size and detection time) to Firebase Realtime Database
    pothole_data = {
        'frame_url': frame_url,
        'estimated_size_cm2': estimated_size,
        'detection_time': detection_time
    }

    # Push data to Firebase Realtime Database
    db.reference('pothole-detections').push(pothole_data)

# Initialize Firebase
cred = credentials.Certificate("potholehunter-175ce-firebase-adminsdk-phcjq-0b2986a9ea.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'potholehunter-175ce.appspot.com',  # Correct Firebase Storage Bucket
    'databaseURL': 'https://potholehunter-175ce-default-rtdb.asia-southeast1.firebasedatabase.app/'  # Correct Firebase Realtime Database URL
})

bucket = storage.bucket()

# Load the YOLOv8 model
model = YOLO('pothole (1).pt')

# Open the video file
video_path = "video.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
frame_count = 0
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Extract results for visualization
        annotated_frame = frame.copy()

        for result in results:
            # Extract bounding box coordinates
            boxes = result.boxes.xyxy  # x_min, y_min, x_max, y_max format

            # Convert tensor to a list for JSON serialization
            boxes = boxes.tolist()

            for box in boxes:
                x_min, y_min, x_max, y_max = box

                # Calculate the width and height of the bounding box
                width = x_max - x_min
                height = y_max - y_min

                # Estimate the size (area) of the pothole
                area = width * height

                # Convert area to a more meaningful measurement if needed (e.g., pixels to cm² or m²)
                # Placeholder conversion factor; adjust based on actual calibration
                conversion_factor = 0.0264
                estimated_size = area * conversion_factor

                # Draw the bounding box on the frame
                cv2.rectangle(annotated_frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

                # Display the estimated size on the frame
                text = f"Size: {estimated_size:.2f} cm²"
                cv2.putText(annotated_frame, text, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Get current date and time
                detection_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # Save frame to Firebase in a separate thread
                success, buffer = cv2.imencode('.jpg', annotated_frame)
                frame_bytes = np.array(buffer).tobytes()

                # Start a new thread to handle the uploading
                threading.Thread(target=upload_frame_to_firebase, args=(frame_bytes, estimated_size, detection_time)).start()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_count += 1
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Create directory for saved faces if it doesn't exist
saved_faces_dir = "saved_faces"
if not os.path.exists(saved_faces_dir):
    os.makedirs(saved_faces_dir)

# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file("obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load a second sample picture and learn how to recognize it.
biden_image = face_recognition.load_image_file("biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding
]
known_face_names = [
    "Barack Obama",
    "Joe Biden"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# Variables for LED indicators and face tracking
led_state = "off"  # "off", "red", "green"
led_timer = 0
led_duration = 30  # frames to show LED
detected_faces = set()  # Track detected face encodings
saved_faces = set()  # Track saved face encodings

def draw_led_indicator(frame, color):
    """Draw LED indicator in bottom right corner"""
    height, width = frame.shape[:2]
    led_size = 30
    led_x = width - led_size - 20
    led_y = height - led_size - 20
    
    if color == "red":
        cv2.circle(frame, (led_x + led_size//2, led_y + led_size//2), led_size//2, (0, 0, 255), -1)
    elif color == "green":
        cv2.circle(frame, (led_x + led_size//2, led_y + led_size//2), led_size//2, (0, 255, 0), -1)
    
    # Add border
    cv2.circle(frame, (led_x + led_size//2, led_y + led_size//2), led_size//2, (255, 255, 255), 2)

def save_current_face(frame, face_locations, face_encodings):
    """Save the current face to disk"""
    if len(face_locations) > 0 and len(face_encodings) > 0:
        # Get the first detected face
        top, right, bottom, left = face_locations[0]
        face_encoding = face_encodings[0]
        
        # Scale back up face location
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        # Extract face region
        face_image = frame[top:bottom, left:right]
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"face_{timestamp}.jpg"
        filepath = os.path.join(saved_faces_dir, filename)
        
        # Save the face image
        cv2.imwrite(filepath, face_image)
        
        # Add to known faces
        known_face_encodings.append(face_encoding)
        known_face_names.append(f"Saved_Face_{len(known_face_names)}")
        
        # Mark as saved
        saved_faces.add(tuple(face_encoding))
        
        print(f"Face saved as {filepath}")
        return True
    return False

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Only process every other frame of video to save time
    if process_this_frame:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        new_face_detected = False
        
        for face_encoding in face_encodings:
            # Check if this is a new face (not previously detected)
            face_tuple = tuple(face_encoding)
            if face_tuple not in detected_faces:
                detected_faces.add(face_tuple)
                new_face_detected = True
                if face_tuple not in saved_faces:
                    led_state = "red"
                    led_timer = led_duration
            
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Handle LED indicator
    if led_timer > 0:
        led_timer -= 1
        if led_state == "red":
            draw_led_indicator(frame, "red")
        elif led_state == "green":
            draw_led_indicator(frame, "green")
    else:
        led_state = "off"

    # Add instructions text
    cv2.putText(frame, "Press 's' to save face, 'q' to quit", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # Save current face
        if save_current_face(frame, face_locations, face_encodings):
            led_state = "green"
            led_timer = led_duration

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

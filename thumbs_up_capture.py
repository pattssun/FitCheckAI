import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime

# Create screenshots directory if it doesn't exist
if not os.path.exists('screenshots'):
    os.makedirs('screenshots')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Variables for screenshot animation
is_screenshot_animation = False
animation_frames = 0
ANIMATION_DURATION = 10  # Frames to show the animation
screenshot_taken = False
last_screenshot_time = datetime.now()
COOLDOWN_TIME = 2  # Seconds between screenshots to avoid duplicates

# Thumbs up detection function
def is_thumbs_up(hand_landmarks):
    # Get thumb tip and other fingertips
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    # Get middle positions of fingers (middle phalanges)
    index_mid = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_mid = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_mid = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_mid = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
    
    # Check if thumb is extended up and other fingers are folded
    # Thumb is up if y-coordinate is significantly lower than other fingertips
    thumb_is_up = thumb_tip.y < index_tip.y and thumb_tip.y < middle_tip.y and thumb_tip.y < ring_tip.y and thumb_tip.y < pinky_tip.y
    
    # Other fingers are folded if fingertips are below their middle points
    fingers_folded = (index_tip.y > index_mid.y and 
                     middle_tip.y > middle_mid.y and 
                     ring_tip.y > ring_mid.y and 
                     pinky_tip.y > pinky_mid.y)
    
    return thumb_is_up and fingers_folded

print("Starting webcam capture. Show a thumbs up to take a screenshot!")
print("Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to read frame from webcam.")
        break
    
    # Flip the frame horizontally for a more intuitive mirror view
    frame = cv2.flip(frame, 1)
    
    # Convert the BGR image to RGB and process with MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    # Initialize thumbs_up status
    thumbs_up_detected = False
    
    # If hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
            # Check for thumbs up gesture
            if is_thumbs_up(hand_landmarks):
                thumbs_up_detected = True
                current_time = datetime.now()
                # Take screenshot if cooldown has passed
                if (current_time - last_screenshot_time).total_seconds() >= COOLDOWN_TIME and not screenshot_taken:
                    # Generate filename with timestamp
                    timestamp = current_time.strftime("%Y%m%d_%H%M%S")
                    filename = f"screenshots/thumbs_up_{timestamp}.jpg"
                    
                    # Save the screenshot
                    cv2.imwrite(filename, frame)
                    print(f"Screenshot saved: {filename}")
                    
                    # Start animation and update status
                    is_screenshot_animation = True
                    animation_frames = 0
                    screenshot_taken = True
                    last_screenshot_time = current_time
    
    # Reset screenshot taken status if no thumbs up is detected
    if not thumbs_up_detected:
        screenshot_taken = False
    
    # Add screenshot animation effect
    if is_screenshot_animation:
        # Flash effect
        animation_frames += 1
        if animation_frames <= ANIMATION_DURATION:
            # Create white border that fades out
            opacity = 1 - (animation_frames / ANIMATION_DURATION)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (255, 255, 255), 20)
            cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)
            
            # Add camera shutter icon in the center
            icon_size = 100
            center_x = frame.shape[1] // 2 - icon_size // 2
            center_y = frame.shape[0] // 2 - icon_size // 2
            cv2.circle(frame, (frame.shape[1] // 2, frame.shape[0] // 2), icon_size, (255, 255, 255), 3)
            cv2.circle(frame, (frame.shape[1] // 2, frame.shape[0] // 2), icon_size - 10, (255, 255, 255), 2)
        else:
            is_screenshot_animation = False
    
    # Add instructions
    cv2.putText(frame, "Show thumbs up to take a photo", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('Thumbs Up Screenshot App', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close() 
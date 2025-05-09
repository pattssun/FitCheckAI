import cv2
import mediapipe as mp
import numpy as np
import os
import pyautogui
import tkinter as tk
from PIL import Image, ImageTk
from datetime import datetime

# Create screenshots directory if it doesn't exist
if not os.path.exists('screenshots'):
    os.makedirs('screenshots')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,  # Lower threshold for better detection
    min_tracking_confidence=0.5)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Variables for screenshot tracking
screenshot_taken = False
last_screenshot_time = datetime.now()
COOLDOWN_TIME = 1  # Reduced cooldown time for more responsive detection

# Create notification window
root = tk.Tk()
root.title("Screenshot Notification")
root.geometry("300x150")
root.configure(bg="#333333")
root.withdraw()  # Hide window initially

notification_label = tk.Label(root, text="Screenshot Captured!", font=("Arial", 14), 
                              fg="white", bg="#333333", pady=20)
notification_label.pack()

filename_label = tk.Label(root, text="", font=("Arial", 10), fg="#aaaaaa", bg="#333333")
filename_label.pack()

# Thumbs up detection function - improved for better reliability
def is_thumbs_up(hand_landmarks):
    # Get thumb tip and other fingertips
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    
    # Get middle positions of fingers (middle phalanges)
    index_mid = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_mid = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_mid = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_mid = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
    
    # More reliable thumbs up detection logic
    # Check if thumb is pointing upwards relative to wrist
    thumb_is_up = thumb_tip.y < wrist.y
    
    # Other fingers are folded if fingertips are closer to wrist than their middle joints
    index_folded = index_tip.y > index_mid.y
    middle_folded = middle_tip.y > middle_mid.y
    ring_folded = ring_tip.y > ring_mid.y
    pinky_folded = pinky_tip.y > pinky_mid.y
    
    # At least 3 other fingers should be folded for a thumbs up
    fingers_folded = sum([index_folded, middle_folded, ring_folded, pinky_folded]) >= 3
    
    return thumb_is_up and fingers_folded

def show_notification(filename):
    """Show notification window with the captured screenshot filename"""
    filename_label.config(text=f"Saved as: {os.path.basename(filename)}")
    
    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")
    
    root.deiconify()  # Show window
    root.update()
    
    # Auto-hide after 2 seconds
    root.after(2000, root.withdraw)

print("Starting thumbs up detection. Show a thumbs up to take a screenshot of your screen!")
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
    
    # Create a copy of the frame for display
    display_frame = frame.copy()
    
    # Initialize thumbs_up status
    thumbs_up_detected = False
    
    # Draw hand landmarks and check for thumbs up
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks for better visualization
            mp_drawing.draw_landmarks(
                display_frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
            # Check for thumbs up gesture
            if is_thumbs_up(hand_landmarks):
                thumbs_up_detected = True
                # Mark the detection on the frame
                cv2.putText(display_frame, "THUMBS UP!", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                current_time = datetime.now()
                # Take screenshot if cooldown has passed
                if (current_time - last_screenshot_time).total_seconds() >= COOLDOWN_TIME and not screenshot_taken:
                    # Generate filename with timestamp
                    timestamp = current_time.strftime("%Y%m%d_%H%M%S")
                    filename = f"screenshots/browser_screenshot_{timestamp}.png"
                    
                    # Take screenshot of the entire screen
                    screenshot = pyautogui.screenshot()
                    screenshot.save(filename)
                    print(f"Screenshot saved: {filename}")
                    
                    # Show notification
                    show_notification(filename)
                    
                    # Update status
                    screenshot_taken = True
                    last_screenshot_time = current_time
    
    # Add status information to the frame
    cv2.putText(display_frame, "Show thumbs up to take a screenshot", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    if screenshot_taken:
        cooldown_remaining = max(0, COOLDOWN_TIME - (datetime.now() - last_screenshot_time).total_seconds())
        cooldown_text = f"Cooldown: {cooldown_remaining:.1f}s"
        cv2.putText(display_frame, cooldown_text, (10, display_frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    
    # Reset screenshot taken status if no thumbs up is detected
    if not thumbs_up_detected:
        screenshot_taken = False
    
    # Display the webcam feed
    cv2.imshow("Webcam Feed - Thumbs Up Detection", display_frame)
    
    # Check for 'q' key press to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Process tkinter events
    root.update_idletasks()
    root.update()

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()
root.destroy() 
import cv2
import mediapipe as mp
import numpy as np
import os
import pyautogui
import tkinter as tk
from PIL import Image, ImageTk
from datetime import datetime
import time

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
    min_detection_confidence=0.7,  # Increased confidence threshold
    min_tracking_confidence=0.7)   # Increased tracking threshold

# Get screen dimensions
screen_width, screen_height = pyautogui.size()
# Calculate the right third of the screen
region_width = screen_width // 4
region_x = screen_width - region_width
region = (region_x, 0, region_width, screen_height)

# Variables for screenshot tracking
screenshot_taken = False
last_screenshot_time = datetime.now()
COOLDOWN_TIME = 3  # Increased cooldown time
countdown_active = False
countdown_start = 0
COUNTDOWN_DURATION = 3  # 3 second countdown

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
    
    # All other fingers should be folded for a thumbs up
    fingers_folded = all([index_folded, middle_folded, ring_folded, pinky_folded])
    
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

print("Starting thumbs up detection. Show a thumbs up in the right third of your screen to take a screenshot!")
print("Press 'q' to quit.")

while True:
    # Capture the right third of the screen
    frame = np.array(pyautogui.screenshot(region=region))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
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
            if is_thumbs_up(hand_landmarks) and not countdown_active and not screenshot_taken:
                countdown_active = True
                countdown_start = time.time()
                thumbs_up_detected = True
    
    # Update countdown if active
    if countdown_active:
        current_time = time.time()
        elapsed = current_time - countdown_start
        remaining = max(0, COUNTDOWN_DURATION - elapsed)
        
        if remaining > 0:
            # Display countdown
            countdown_text = f"Taking screenshot in: {int(remaining) + 1}"
            cv2.putText(display_frame, countdown_text, (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # Take screenshot when countdown reaches 0
            if not screenshot_taken:
                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"screenshots/iphone_mirror_{timestamp}.png"
                
                # Take screenshot of only the right third of the screen
                screenshot = pyautogui.screenshot(region=region)
                screenshot.save(filename)
                print(f"Screenshot saved: {filename}")
                
                # Show notification
                show_notification(filename)
                
                # Update status
                screenshot_taken = True
                last_screenshot_time = datetime.now()
                countdown_active = False
    
    # Reset screenshot taken status after cooldown
    if screenshot_taken:
        if (datetime.now() - last_screenshot_time).total_seconds() >= COOLDOWN_TIME:
            screenshot_taken = False
    
    # Add status information to the frame
    if not countdown_active:
        cv2.putText(display_frame, "Show thumbs up to start countdown", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display the captured region
    cv2.imshow("iPhone Mirroring - Thumbs Up Detection", display_frame)
    
    # Check for 'q' key press to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Process tkinter events
    root.update_idletasks()
    root.update()

# Release resources
cv2.destroyAllWindows()
hands.close()
root.destroy() 
# FitCheckAI

## Thumbs Up Screenshot Tool

This application uses your webcam to detect thumbs up gestures and automatically capture screenshots when detected.

### Features
- Real-time hand gesture detection using MediaPipe
- Captures screenshots when thumbs up gesture is detected
- Visual feedback with animation when screenshot is taken
- Screenshots saved with timestamps in the "screenshots" folder

### Setup
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the application:
   ```
   python thumbs_up_capture.py
   ```

3. Usage:
   - Show a thumbs up gesture to take a screenshot
   - Press 'q' to quit the application

### Requirements
- Python 3.7+
- Webcam access
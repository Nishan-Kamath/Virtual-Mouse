
---

# 🖱️ Virtual AI Mouse

A virtual mouse application using hand gestures to control the cursor and perform actions such as clicking, powered by MediaPipe and OpenCV. This project uses hand-tracking technology to allow users to move their mouse and perform clicks through simple hand gestures.

### 🔑 Features:
- **Hand Gesture Tracking**: Uses index and middle finger positions to control cursor movement.
- **Smooth Mouse Control**: Implements a smoothing algorithm for fluid cursor movement.
- **Click Detection**: Automatically clicks when the index and middle fingers come close together.
- **Perspective Mapping**: Maps hand movements in the webcam feed to screen coordinates for precise control.
- **Real-Time Processing**: Utilizes real-time video capture for a responsive experience.

### 🛠️ Technologies Used:
- **OpenCV**: For video capture and processing.
- **MediaPipe**: For hand landmark detection and gesture recognition.
- **PyAutoGUI**: To control the mouse and simulate clicks.
- **NumPy**: For mathematical operations and perspective transformations.

### 🎯 How It Works:
1. **Hand Detection**: MediaPipe detects the landmarks of the index and middle fingers.
2. **Mapping to Screen**: Using a perspective transformation, the detected finger movements are mapped to your screen dimensions.
3. **Cursor Movement**: PyAutoGUI moves the cursor to the transformed coordinates, creating a virtual mouse experience.
4. **Click Action**: When the distance between the index and middle fingers is below a certain threshold, the mouse click action is triggered.

### ⚙️ Setup Instructions:
1. Install the required dependencies:
    ```bash
    pip install opencv-python mediapipe pyautogui numpy
    ```
2. Run the script:
    ```bash
    python virtual_mouse.py
    ```

3. Use your webcam to move the cursor using hand gestures. Press `q` to exit the program.

---

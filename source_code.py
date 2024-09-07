import cv2
import mediapipe as mp
import pyautogui
import numpy as np

pyautogui.FAILSAFE = False
# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Set up PyAutoGUI screen dimensions (adjust based on your screen resolution)
screen_width, screen_height = pyautogui.size()

# Initialize variables for smoothing
smooth_factor = 0.2
prev_x, prev_y = 0, 0

# Points for perspective transform
# You need to adjust these points based on your camera and screen setup
src_points = np.float32([[100, 100], [540, 100], [100, 380], [540, 380]])  # From camera
dst_points = np.float32([[0, 0], [screen_width, 0], [0, screen_height], [screen_width, screen_height]])  # To screen

# Calculate perspective transform matrix
M = cv2.getPerspectiveTransform(src_points, dst_points)

# Video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Read a frame from the webcam
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    # Convert the frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe
    results = hands.process(rgb_frame)

    # Extract hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmarks for the index and middle fingers (adjust as needed)
            index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            # Map hand landmarks to screen coordinates using perspective transform
            index_finger_point = np.array([[index_finger.x * frame.shape[1], index_finger.y * frame.shape[0]]], dtype='float32')
            index_finger_point = np.array([index_finger_point])
            transformed_point = cv2.perspectiveTransform(index_finger_point, M)
            x, y = int(transformed_point[0][0][0]), int(transformed_point[0][0][1])

            # Smooth the cursor movement
            x = int(prev_x * (1 - smooth_factor) + x * smooth_factor)
            y = int(prev_y * (1 - smooth_factor) + y * smooth_factor)
            prev_x, prev_y = x, y

            # Move the mouse to the mapped coordinates using PyAutoGUI
            pyautogui.moveTo(x, y, duration=0.1)

            # Optionally, click the mouse when fingers are close
            distance = np.sqrt((index_finger.x - middle_finger.x) ** 2 + (index_finger.y - middle_finger.y) ** 2)
            if distance < 0.05:
                pyautogui.click()

    cv2.imshow("Virtual AI Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the OpenCV window
cap.release()
cv2.destroyAllWindows()

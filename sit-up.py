import cv2
import mediapipe as mp
import numpy as np
import time
import os
import simpleaudio as sa
from datetime import datetime

def play_rep_audio(count):
    path = f'audio/{count}.wav'
    try:
        wave_obj = sa.WaveObject.from_wave_file(path)
        play_obj = wave_obj.play()
        # No need to wait; it's non-blocking
    except Exception as e:
        print(f"Audio error for {count}: {e}")

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Sit-up detection variables
counter = 0
stage = "up"
frame_threshold = 1
up_frames = 0
down_frames = 0
last_rep_time = 0
min_time_between_reps = 0.3  # seconds
base_nose_y = None
base_shoulder_y = None

cap = cv2.VideoCapture(0)

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
recording_name = f"situp_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
out = cv2.VideoWriter(recording_name, fourcc, 20.0, (640, 480))

with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Key body points
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                    landmarks[mp_pose.PoseLandmark.NOSE.value].y]

            # Calculate average shoulder position
            shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2

            # Angle between shoulder, hip, and knee
            hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)

            # Save base (down) position
            if hip_angle > 110 or shoulder_y > 0.65:
                down_frames += 1
                up_frames = 0
                if down_frames >= frame_threshold:
                    stage = "down"
                    base_nose_y = nose[1]
                    base_shoulder_y = shoulder_y

            # Sit-up detection (count partial sit-ups, avoid standing)
            if stage == "down" and (nose[1] < base_nose_y - 0.01 or shoulder_y < base_shoulder_y - 0.0q1) and hip_angle < 170:
                up_frames += 1
                down_frames = 0
                current_time = time.time()
                if up_frames >= frame_threshold and stage == "down":
                    if current_time - last_rep_time > min_time_between_reps:
                        counter += 1
                        stage = "up"
                        last_rep_time = current_time
                         # Play count audio
                        play_rep_audio(counter)

            # Display hip angle, stage, and counter
            cv2.putText(image, f"Hip Angle: {int(hip_angle)}", (50, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            # cv2.putText(image, f"Stage: {stage}", (50, 100),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(image, f"Sit-Ups: {counter}", (50, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        except:
            pass

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow('Sit-Up Counter', image)
        
        # Write frame to video
        out.write(image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()
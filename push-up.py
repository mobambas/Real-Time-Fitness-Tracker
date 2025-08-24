import cv2
import mediapipe as mp
import numpy as np
import time
import os
import simpleaudio as sa

def play_rep_audio(count):
    path = f'audio/{count}.wav'
    try:
        wave_obj = sa.WaveObject.from_wave_file(path)
        play_obj = wave_obj.play()
    except Exception as e:
        print(f"Audio error for {count}: {e}")

def play_wrong_audio():
    path = 'audio/wrong.wav'
    try:
        wave_obj = sa.WaveObject.from_wave_file(path)
        play_obj = wave_obj.play()
    except Exception as e:
        print(f"Audio error for wrong: {e}")

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Repetition counter variables
counter = 0
stage = None
down_frames = 0
up_frames = 0
partial_down_frames = 0
frame_threshold = 1  # Number of consistent frames to confirm a position

# Time buffer to avoid double-counting
last_rep_time = 0
min_time_between_reps = 0.25  # seconds

# Wrong rep alert variables
wrong_alert_start = -1
partial_rep_detected = False  # Track if a partial rep was started

# Video capture
cap = cv2.VideoCapture(0)

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('pushup_session.mp4', fourcc, 20.0, (640, 480))

# Calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Pose detection
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        # Convert to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Left arm
            l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Right arm
            r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Hip and knee for body alignment
            l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

            # Use arm closer to camera
            l_elbow_z = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z
            r_elbow_z = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z
            use_left = l_elbow_z < r_elbow_z

            if use_left:
                angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
                elbow_coords = l_elbow
                arm_side = "Left"
                back_angle = calculate_angle(l_shoulder, l_hip, l_knee)
            else:
                angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
                elbow_coords = r_elbow
                arm_side = "Right"
                back_angle = calculate_angle(r_shoulder, r_hip, r_knee)

            # Display angles and arm used
            cv2.putText(image, f'Elbow: {int(angle)}', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, f'Back: {int(back_angle)}', (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 255, 255), 2)
            cv2.putText(image, f'Using: {arm_side} Arm', (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Push-up detection with frame buffer and minimal time delay
            current_time = time.time()
            if angle > 160 and back_angle > 160:
                up_frames += 1
                down_frames = 0
                partial_down_frames = 0
                if up_frames > frame_threshold:
                    # If returning to "up" after a partial rep without a good rep, trigger wrong alert
                    if partial_rep_detected and current_time - last_rep_time > min_time_between_reps:
                        wrong_alert_start = current_time
                        play_wrong_audio()
                        partial_rep_detected = False
                    stage = "up"

            elif angle < 100 and back_angle > 160:
                down_frames += 1
                up_frames = 0
                partial_down_frames = 0
                partial_rep_detected = False  # Reset partial rep if good rep is detected
                if down_frames > frame_threshold and stage == "up":
                    if current_time - last_rep_time > min_time_between_reps:
                        stage = "down"
                        counter += 1
                        last_rep_time = current_time
                        play_rep_audio(counter)

            elif 110 < angle < 150 and back_angle > 160 and stage == "up":
                partial_down_frames += 1
                down_frames = 0
                up_frames = 0
                if partial_down_frames > frame_threshold:
                    partial_rep_detected = True  # Mark that a partial rep started

            # Display push-up count
            cv2.putText(image, f'Push-Ups: {counter}', (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            # Display wrong rep alert pop-up if within 2 seconds
            if wrong_alert_start > 0 and current_time - wrong_alert_start < 2:
                alert_text = "Wrong Rep! Go Deeper!"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                thickness = 3
                text_size = cv2.getTextSize(alert_text, font, font_scale, thickness)[0]
                rect_start = (int(640 / 2 - text_size[0] / 2 - 20), int(480 / 2 - text_size[1] / 2 - 20))
                rect_end = (int(640 / 2 + text_size[0] / 2 + 20), int(480 / 2 + text_size[1] / 2 + 20))
                cv2.rectangle(image, rect_start, rect_end, (0, 0, 255), -1)  # Red background
                cv2.putText(image, alert_text,
                            (int(640 / 2 - text_size[0] / 2), int(480 / 2 + text_size[1] / 2)),
                            font, font_scale, (255, 255, 255), thickness)

        except:
            pass

        # Draw pose landmarks
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
        )

        # Show and save
        cv2.imshow('Push-Up Counter', image)
        out.write(image)

        time.sleep(0.03)  # ~30 FPS

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()

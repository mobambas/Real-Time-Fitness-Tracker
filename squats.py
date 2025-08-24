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
        # No need to wait; it's non-blocking
    except Exception as e:
        print(f"Audio error for {count}: {e}")

def play_wrong_audio():
    path = 'audio/wrong.wav'
    try:
        wave_obj = sa.WaveObject.from_wave_file(path)
        play_obj = wave_obj.play()
        # No need to wait; it's non-blocking
    except Exception as e:
        print(f"Audio error for wrong: {e}")

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

# Squat detection variables
counter = 0
stage = "up"
frame_threshold = 1
up_frames = 0
down_frames = 0
partial_down_frames = 0
last_rep_time = 0
min_time_between_reps = 0.4  # seconds
partial_rep_detected = False
wrong_alert_start = -1

cap = cv2.VideoCapture(0)

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('squat_session.mp4', fourcc, 20.0, (640, 480))

with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Get both legs
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            # Calculate both knee angles
            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

            # Use the leg with the smaller angle (more bent)
            knee_angle = min(left_knee_angle, right_knee_angle)

            current_time = time.time()

            # Squat logic
            if knee_angle > 160:
                up_frames += 1
                down_frames = 0
                partial_down_frames = 0
                if up_frames >= frame_threshold:
                    # If returning to "up" after a partial rep without a good rep, trigger wrong alert
                    if partial_rep_detected and current_time - last_rep_time > min_time_between_reps:
                        wrong_alert_start = current_time
                        play_wrong_audio()
                        partial_rep_detected = False
                    stage = "up"

            elif knee_angle < 120:
                down_frames += 1
                up_frames = 0
                partial_down_frames = 0
                partial_rep_detected = False  # Reset partial rep if good rep is detected
                if down_frames >= frame_threshold and stage == "up":
                    if current_time - last_rep_time > min_time_between_reps:
                        counter += 1
                        stage = "down"
                        last_rep_time = current_time
                        play_rep_audio(counter)

            elif 135 < knee_angle < 160 and stage == "up":
                partial_down_frames += 1
                down_frames = 0
                up_frames = 0
                if partial_down_frames >= frame_threshold:
                    partial_rep_detected = True  # Mark that a partial rep started

            # Show both knee angles and squat count
            cv2.putText(image, f"L: {int(left_knee_angle)}  R: {int(right_knee_angle)}", (50, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(image, f"Squats: {counter}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            # Display wrong rep alert pop-up if within 2 seconds (with improved styling)
            if wrong_alert_start > 0 and current_time - wrong_alert_start < 2:
                alert_text = "Wrong Rep! Go Deeper!"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                thickness = 3
                text_size = cv2.getTextSize(alert_text, font, font_scale, thickness)[0]
                text_x = int(640 / 2 - text_size[0] / 2)
                text_y = int(480 / 2 + text_size[1] / 2)
                rect_start = (text_x - 20, int(480 / 2 - text_size[1] / 2 - 20))
                rect_end = (text_x + text_size[0] + 20, int(480 / 2 + text_size[1] / 2 + 20))

                # Semi-transparent red background
                overlay = image.copy()
                cv2.rectangle(overlay, rect_start, rect_end, (0, 0, 255), -1)
                alpha = 0.6  # Transparency factor
                cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

                # Text with shadow for better visibility
                shadow_offset = 2
                cv2.putText(image, alert_text,
                            (text_x + shadow_offset, text_y + shadow_offset),
                            font, font_scale, (0, 0, 0), thickness + 1)
                cv2.putText(image, alert_text,
                            (text_x, text_y),
                            font, font_scale, (255, 255, 255), thickness)

        except:
            pass

        # Draw pose
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow('Squat Counter', image)

        # Save to video
        out.write(image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()

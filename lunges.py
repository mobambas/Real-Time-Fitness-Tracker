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

# Pose Setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Rep Counter
counter_left = 0
counter_right = 0
stage_left = "up"
stage_right = "up"
frame_threshold = 5  # For state transitions
min_left_knee_angle = 180.0  # Track minimum angle in cycle
min_right_knee_angle = 180.0
min_time_between_reps = 1.5
min_time_between_wrong_alerts = 2.0
last_rep_time_left = 0
last_rep_time_right = 0
last_wrong_alert_time = -2.0  # Allow first wrong rep immediately
left_down_frames = 0
right_down_frames = 0
left_up_frames = 0
right_up_frames = 0
wrong_alert_start = -1

def calculate_angle(a, b, c):
    """Returns angle between 3 points (a, b, c) in degrees"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180 else 360 - angle

cap = cv2.VideoCapture(0)

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('lunges_session.mp4', fourcc, 20.0, (640, 480))

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
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

        if results.pose_landmarks:
            try:
                landmarks = results.pose_landmarks.landmark

                # Left leg keypoints
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                # Right leg keypoints
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                # Calculate angles
                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

                # Get z and y for depth and height
                left_knee_z = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z
                right_knee_z = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z
                left_knee_y = left_knee[1]
                right_knee_y = right_knee[1]

                current_time = time.time()

                forward_leg = "None"
                # Detect forward leg using knee z-position (larger z = farther = forward)
                if left_knee_z > right_knee_z:  # left farther, left forward
                    forward_leg = "Left"
                    # Nested y-coordinate check for left leg forward
                    if left_knee_y < right_knee_y - 0.03:  # left leg forward
                        if stage_left == "up":
                            if left_knee_angle < 160:
                                left_down_frames += 1
                                min_left_knee_angle = min(min_left_knee_angle, left_knee_angle)
                                if left_down_frames >= frame_threshold:
                                    stage_left = "down"
                                    left_down_frames = 0
                            else:
                                left_down_frames = 0
                        elif stage_left == "down":
                            min_left_knee_angle = min(min_left_knee_angle, left_knee_angle)
                            if left_knee_angle > 160:
                                left_up_frames += 1
                                if left_up_frames >= frame_threshold:
                                    if current_time - last_rep_time_left > min_time_between_reps:
                                        if min_left_knee_angle < 100:
                                            counter_left += 1
                                            last_rep_time_left = current_time
                                            play_rep_audio(counter_left)
                                        elif 110 <= min_left_knee_angle < 150:
                                            if current_time - last_wrong_alert_time > min_time_between_wrong_alerts:
                                                wrong_alert_start = current_time
                                                last_wrong_alert_time = current_time
                                                play_wrong_audio()
                                    stage_left = "up"
                                    min_left_knee_angle = 180.0
                                    left_up_frames = 0
                            else:
                                left_up_frames = 0
                    # Nested y-coordinate check for right leg forward (in case z is misleading)
                    elif right_knee_y < left_knee_y - 0.03:  # right leg forward
                        if stage_right == "up":
                            if right_knee_angle < 160:
                                right_down_frames += 1
                                min_right_knee_angle = min(min_right_knee_angle, right_knee_angle)
                                if right_down_frames >= frame_threshold:
                                    stage_right = "down"
                                    right_down_frames = 0
                            else:
                                right_down_frames = 0
                        elif stage_right == "down":
                            min_right_knee_angle = min(min_right_knee_angle, right_knee_angle)
                            if right_knee_angle > 160:
                                right_up_frames += 1
                                if right_up_frames >= frame_threshold:
                                    if current_time - last_rep_time_right > min_time_between_reps:
                                        if min_right_knee_angle < 100:
                                            counter_right += 1
                                            last_rep_time_right = current_time
                                            play_rep_audio(counter_right)
                                        elif 110 <= min_right_knee_angle < 150:
                                            if current_time - last_wrong_alert_time > min_time_between_wrong_alerts:
                                                wrong_alert_start = current_time
                                                last_wrong_alert_time = current_time
                                                play_wrong_audio()
                                    stage_right = "up"
                                    min_right_knee_angle = 180.0
                                    right_up_frames = 0
                            else:
                                right_up_frames = 0
                else:  # right farther, right forward
                    forward_leg = "Right"
                    # Nested y-coordinate check for right leg forward
                    if right_knee_y < left_knee_y - 0.03:  # right leg forward
                        if stage_right == "up":
                            if right_knee_angle < 160:
                                right_down_frames += 1
                                min_right_knee_angle = min(min_right_knee_angle, right_knee_angle)
                                if right_down_frames >= frame_threshold:
                                    stage_right = "down"
                                    right_down_frames = 0
                            else:
                                right_down_frames = 0
                        elif stage_right == "down":
                            min_right_knee_angle = min(min_right_knee_angle, right_knee_angle)
                            if right_knee_angle > 160:
                                right_up_frames += 1
                                if right_up_frames >= frame_threshold:
                                    if current_time - last_rep_time_right > min_time_between_reps:
                                        if min_right_knee_angle < 100:
                                            counter_right += 1
                                            last_rep_time_right = current_time
                                            play_rep_audio(counter_right)
                                        elif 110 <= min_right_knee_angle < 150:
                                            if current_time - last_wrong_alert_time > min_time_between_wrong_alerts:
                                                wrong_alert_start = current_time
                                                last_wrong_alert_time = current_time
                                                play_wrong_audio()
                                    stage_right = "up"
                                    min_right_knee_angle = 180.0
                                    right_up_frames = 0
                            else:
                                right_up_frames = 0
                    # Nested y-coordinate check for left leg forward (in case z is misleading)
                    elif left_knee_y < right_knee_y - 0.03:  # left leg forward
                        if stage_left == "up":
                            if left_knee_angle < 160:
                                left_down_frames += 1
                                min_left_knee_angle = min(min_left_knee_angle, left_knee_angle)
                                if left_down_frames >= frame_threshold:
                                    stage_left = "down"
                                    left_down_frames = 0
                            else:
                                left_down_frames = 0
                        elif stage_left == "down":
                            min_left_knee_angle = min(min_left_knee_angle, left_knee_angle)
                            if left_knee_angle > 160:
                                left_up_frames += 1
                                if left_up_frames >= frame_threshold:
                                    if current_time - last_rep_time_left > min_time_between_reps:
                                        if min_left_knee_angle < 100:
                                            counter_left += 1
                                            last_rep_time_left = current_time
                                            play_rep_audio(counter_left)
                                        elif 110 <= min_left_knee_angle < 150:
                                            if current_time - last_wrong_alert_time > min_time_between_wrong_alerts:
                                                wrong_alert_start = current_time
                                                last_wrong_alert_time = current_time
                                                play_wrong_audio()
                                    stage_left = "up"
                                    min_left_knee_angle = 180.0
                                    left_up_frames = 0
                            else:
                                left_up_frames = 0

                # Draw counts and angles
                cv2.putText(image, f"Left Lunges: {counter_left}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(image, f"Right Lunges: {counter_right}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f"L: {int(left_knee_angle)} R: {int(right_knee_angle)}", (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(image, f"Forward: {forward_leg}", (10, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # Display wrong rep alert pop-up if within 2 seconds
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
                    alpha = 0.6
                    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

                    # Text with shadow
                    shadow_offset = 2
                    cv2.putText(image, alert_text,
                                (text_x + shadow_offset, text_y + shadow_offset),
                                font, font_scale, (0, 0, 0), thickness + 1)
                    cv2.putText(image, alert_text,
                                (text_x, text_y),
                                font, font_scale, (255, 255, 255), thickness)

                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            except Exception as e:
                print("Error inside landmarks block:", e)

        else:
            cv2.putText(image, "No person detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Lunges Counter", image)

        # Save to video
        out.write(image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()

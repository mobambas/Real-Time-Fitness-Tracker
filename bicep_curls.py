import cv2
import mediapipe as mp
import numpy as np
import time
import simpleaudio as sa
import random
from datetime import datetime


def play_rep_audio(count: int) -> None:
    """Play numbered rep audio if available."""
    path = f'audio/{count}.wav'
    try:
        wave_obj = sa.WaveObject.from_wave_file(path)
        wave_obj.play()
    except Exception as e:
        print(f"Audio error for {count}: {e}")


def play_wrong_audio() -> None:
    """Play incorrect rep alert."""
    path = 'audio/wrong.wav'
    try:
        wave_obj = sa.WaveObject.from_wave_file(path)
        wave_obj.play()
    except Exception as e:
        print(f"Audio error for wrong: {e}")


# MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    """Returns angle between 3 points (a, b, c) in degrees."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180 else 360 - angle


# UI Constants
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
VIDEO_WIDTH = 800
VIDEO_HEIGHT = 600

# Color scheme (BGR format for OpenCV)
COLOR_DARK_PURPLE = (51, 0, 102)
COLOR_PURPLE = (128, 0, 128)
COLOR_BRIGHT_PURPLE = (255, 0, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)

# Confetti particles
confetti_particles = []


def add_confetti(x, y):
    for _ in range(10):
        confetti_particles.append({
            'x': x,
            'y': y,
            'vx': random.uniform(-3, 3),
            'vy': random.uniform(-5, -1),
            'color': random.choice([COLOR_YELLOW, COLOR_GREEN, COLOR_BRIGHT_PURPLE, COLOR_BLUE]),
            'size': random.randint(3, 8),
            'life': 30
        })


def update_confetti():
    global confetti_particles
    for particle in confetti_particles[:]:
        particle['x'] += particle['vx']
        particle['y'] += particle['vy']
        particle['vy'] += 0.2
        particle['life'] -= 1
        if particle['life'] <= 0:
            confetti_particles.remove(particle)


def draw_confetti(image):
    for particle in confetti_particles:
        cv2.circle(image, (int(particle['x']), int(particle['y'])), particle['size'], particle['color'], -1)


def draw_rounded_rect(image, pt1, pt2, color, thickness=-1, radius=10):
    x1, y1 = pt1
    x2, y2 = pt2

    if thickness == -1:
        cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(image, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(image, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(image, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(image, (x2 - radius, y2 - radius), radius, color, -1)
    else:
        cv2.line(image, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(image, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(image, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(image, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        cv2.ellipse(image, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(image, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(image, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(image, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


def format_time(seconds):
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def get_target_muscles(exercise):
    muscle_map = {
        "squats": "Quadriceps, Glutes",
        "push-ups": "Chest, Triceps, Shoulders",
        "sit-ups": "Abs, Core",
        "lunges": "Quadriceps, Glutes, Hamstrings",
        "bicep curls": "Biceps, Forearms",
        "lat pulldown": "Lats, Biceps, Rear Delts",
        "pull-ups": "Lats, Biceps, Core"
    }
    return muscle_map.get(exercise, "Multiple Muscle Groups")


# Detection variables
counter = 0
incorrect_counter = 0
sets = 1
stage = "down"  # start with arms extended
frame_threshold = 2
up_frames = 0
down_frames = 0
partial_frames = 0
last_rep_time = 0.0
min_time_between_reps = 0.4
wrong_alert_start = -1.0
partial_rep_detected = False
elbow_ref = None  # reference elbow position when arms are extended
elbow_drift_frames = 0
elbow_drift_flag = False
elbow_drift_threshold = 0.04  # normalized distance tolerance for elbow stability
back_bad_frames = 0
back_bad_flag = False
workout_start_time = time.time()
is_paused = False
paused_time = 0.0
pause_start_time = 0.0
wrong_suppress_until = 0.0
last_audio_time = 0.0

# Mouse/button handling
mouse_x = 0
mouse_y = 0
mouse_clicked = False
show_exercise_menu = False
exercise_menu_start_time = 0.0
selected_exercise = "bicep curls"

available_exercises = ["squats", "push-ups", "sit-ups", "lunges", "bicep curls", "lat pulldown", "pull-ups"]

button_coords = {
    'pause': None,
    'change_exercise': None,
    'end_workout': None
}


def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y, mouse_clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_x = x
        mouse_y = y
        mouse_clicked = True


cap = cv2.VideoCapture(0)

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
recording_name = f"bicep_curl_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
out = cv2.VideoWriter(recording_name, fourcc, 20.0, (WINDOW_WIDTH, WINDOW_HEIGHT))

# Create window and set mouse callback
cv2.namedWindow('FitMaster AI - Bicep Curl Counter')
cv2.setMouseCallback('FitMaster AI - Bicep Curl Counter', mouse_callback)

with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        ui_image = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
        ui_image[:] = COLOR_BLACK

        frame_resized = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
        video_image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        video_image.flags.writeable = False
        results = pose.process(video_image)
        video_image.flags.writeable = True
        video_image = cv2.cvtColor(video_image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                video_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            )

        video_x = 20
        video_y = 120

        current_time = time.time()
        if is_paused:
            if pause_start_time == 0:
                pause_start_time = current_time
            workout_elapsed = (pause_start_time - workout_start_time) - paused_time
        else:
            if pause_start_time > 0:
                paused_time += (current_time - pause_start_time)
                pause_start_time = 0
            workout_elapsed = (current_time - workout_start_time) - paused_time

        total_reps_display = counter + incorrect_counter
        if total_reps_display > 0:
            form_quality = max(1, min(5, int(5 * (counter / total_reps_display))))
        else:
            form_quality = 3

        if workout_elapsed > 0:
            reps_per_min = (counter / workout_elapsed) * 60
            intensity = min(100, int(reps_per_min * 5))
        else:
            intensity = 0

        try:
            landmarks = results.pose_landmarks.landmark if results.pose_landmarks else None
            if landmarks:
                l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

                r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

                l_wrist_z = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z
                r_wrist_z = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z
                use_left = l_wrist_z < r_wrist_z

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

                elbow_x = int(elbow_coords[0] * VIDEO_WIDTH)
                elbow_y = int(elbow_coords[1] * VIDEO_HEIGHT)

                cv2.putText(video_image, f"{arm_side} {int(angle)}°",
                            (elbow_x + 20, elbow_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                # Optional: display back angle
                cv2.putText(video_image, f"Back {int(back_angle)}°",
                            (elbow_x + 20, elbow_y + 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                if not is_paused:
                    # Elbow stability: set reference when extended
                    if angle > 150 and not partial_rep_detected:
                        elbow_ref = elbow_coords[:]
                        elbow_drift_frames = 0
                        elbow_drift_flag = False
                    # Track elbow drift during movement
                    if elbow_ref and angle < 150:
                        drift = np.linalg.norm(np.array(elbow_coords) - np.array(elbow_ref))
                        if drift > elbow_drift_threshold:
                            elbow_drift_frames += 1
                        else:
                            elbow_drift_frames = 0
                            elbow_drift_flag = False
                        if elbow_drift_frames >= frame_threshold and not elbow_drift_flag:
                            wrong_alert_start = current_time
                            incorrect_counter += 1
                            elbow_drift_flag = True
                            play_wrong_audio()

                    # Back straightness check
                    if back_angle < 150 and angle < 150:
                        back_bad_frames += 1
                    else:
                        back_bad_frames = 0
                        back_bad_flag = False
                    if back_bad_frames >= frame_threshold and not back_bad_flag and current_time > wrong_suppress_until:
                        wrong_alert_start = current_time
                        incorrect_counter += 1
                        back_bad_flag = True
                        wrong_suppress_until = current_time + 0.6
                        if current_time - last_audio_time > 0.35:
                            play_wrong_audio()
                            last_audio_time = current_time

                    if angle > 150:
                        down_frames += 1
                        up_frames = 0
                        partial_frames = 0
                        if down_frames >= frame_threshold:
                            if partial_rep_detected and current_time - last_rep_time > min_time_between_reps:
                                if current_time > wrong_suppress_until:
                                    wrong_alert_start = current_time
                                    incorrect_counter += 1
                                    wrong_suppress_until = current_time + 0.6
                                    if current_time - last_audio_time > 0.35:
                                        play_wrong_audio()
                                        last_audio_time = current_time
                                partial_rep_detected = False
                            stage = "down"
                    elif angle < 50:
                        up_frames += 1
                        down_frames = 0
                        partial_frames = 0
                        if up_frames >= frame_threshold and stage == "down":
                            if current_time - last_rep_time > min_time_between_reps:
                                counter += 1
                                stage = "up"
                                last_rep_time = current_time
                                wrong_suppress_until = current_time + 0.6
                                if current_time - last_audio_time > 0.35:
                                    play_rep_audio(counter)
                                    last_audio_time = current_time
                                add_confetti(video_x + VIDEO_WIDTH // 2, video_y + VIDEO_HEIGHT // 2)
                                partial_rep_detected = False
                    elif 50 <= angle <= 120 and stage == "down":
                        partial_frames += 1
                        down_frames = 0
                        up_frames = 0
                        if partial_frames >= frame_threshold:
                            partial_rep_detected = True

                # (alert drawing moved after video placement)
        except Exception as e:
            print("Error inside landmarks block:", e)

        update_confetti()
        draw_confetti(ui_image)

        cv2.rectangle(ui_image, (0, 0), (WINDOW_WIDTH, 60), COLOR_DARK_PURPLE, -1)
        cv2.putText(ui_image, "FitMaster AI", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_BRIGHT_PURPLE, 2)

        profile_x = WINDOW_WIDTH - 150
        draw_rounded_rect(ui_image, (profile_x, 15), (WINDOW_WIDTH - 20, 45), COLOR_PURPLE, -1, 15)
        cv2.putText(ui_image, "haro...", (profile_x + 10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)

        header_y = 70
        cv2.putText(ui_image, "Real-time Pose Correction", (20, header_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR_BRIGHT_PURPLE, 3)

        banner_y = header_y + 10
        draw_rounded_rect(ui_image, (20, banner_y + 5), (VIDEO_WIDTH + 20, banner_y + 40), COLOR_YELLOW, -1, 8)
        cv2.putText(ui_image, "Keep elbows tucked and avoid swinging your torso.",
                    (30, banner_y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BLACK, 2)

        card_y = banner_y + 50
        card_width = 160
        card_height = 80
        card_spacing = 15

        card1_x = 20
        total_reps = counter + incorrect_counter
        draw_rounded_rect(ui_image, (card1_x, card_y), (card1_x + card_width, card_y + card_height),
                          COLOR_DARK_PURPLE, -1, 10)
        cv2.putText(ui_image, "Total Reps", (card1_x + 10, card_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)
        cv2.putText(ui_image, str(total_reps), (card1_x + 10, card_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_BRIGHT_PURPLE, 3)

        card2_x = card1_x + card_width + card_spacing
        draw_rounded_rect(ui_image, (card2_x, card_y), (card2_x + card_width, card_y + card_height),
                          COLOR_DARK_PURPLE, -1, 10)
        cv2.putText(ui_image, "Correct Reps", (card2_x + 10, card_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)
        cv2.putText(ui_image, str(counter), (card2_x + 10, card_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_GREEN, 3)

        card3_x = card2_x + card_width + card_spacing
        draw_rounded_rect(ui_image, (card3_x, card_y), (card3_x + card_width, card_y + card_height),
                          COLOR_DARK_PURPLE, -1, 10)
        cv2.putText(ui_image, "Incorrect Reps", (card3_x + 10, card_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)
        cv2.putText(ui_image, str(incorrect_counter), (card3_x + 10, card_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_RED, 3)

        card4_x = card3_x + card_width + card_spacing
        draw_rounded_rect(ui_image, (card4_x, card_y), (card4_x + card_width, card_y + card_height),
                          COLOR_DARK_PURPLE, -1, 10)
        cv2.putText(ui_image, "Total Sets", (card4_x + 10, card_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)
        cv2.putText(ui_image, str(sets), (card4_x + 10, card_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_BRIGHT_PURPLE, 3)

        card5_x = card4_x + card_width + card_spacing
        draw_rounded_rect(ui_image, (card5_x, card_y), (card5_x + card_width, card_y + card_height),
                          COLOR_DARK_PURPLE, -1, 10)
        cv2.putText(ui_image, "Current Exercise", (card5_x + 10, card_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
        exercise_display = selected_exercise.replace('-', ' ')
        cv2.putText(ui_image, exercise_display, (card5_x + 10, card_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_BRIGHT_PURPLE, 2)

        status_x = WINDOW_WIDTH - 345
        status_y = card_y + 15
        cv2.putText(ui_image, "Connection Status: Connected", (status_x, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_GREEN, 2)

        panel_x = VIDEO_WIDTH + 40
        panel_y = card_y

        draw_rounded_rect(ui_image, (panel_x, panel_y), (WINDOW_WIDTH - 20, WINDOW_HEIGHT - 20),
                          COLOR_DARK_PURPLE, -1, 15)

        detail_y = panel_y + 30
        cv2.putText(ui_image, "Exercise Details", (panel_x + 20, detail_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_WHITE, 2)

        detail_y += 40
        cv2.putText(ui_image, "Target Muscles:", (panel_x + 20, detail_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)
        target_muscles = get_target_muscles(selected_exercise)
        cv2.putText(ui_image, target_muscles, (panel_x + 20, detail_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BRIGHT_PURPLE, 2)

        detail_y += 60
        cv2.putText(ui_image, "Rep Breakdown:", (panel_x + 20, detail_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)
        detail_y += 25
        cv2.putText(ui_image, f"Total: {total_reps}", (panel_x + 20, detail_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)
        detail_y += 20
        cv2.putText(ui_image, f"Correct: {counter}", (panel_x + 20, detail_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_GREEN, 2)
        detail_y += 20
        cv2.putText(ui_image, f"Incorrect: {incorrect_counter}", (panel_x + 20, detail_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_RED, 2)
        if total_reps > 0:
            accuracy = int((counter / total_reps) * 100)
            detail_y += 20
            cv2.putText(ui_image, f"Accuracy: {accuracy}%", (panel_x + 20, detail_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_YELLOW, 2)

        detail_y += 60
        cv2.putText(ui_image, "Intensity:", (panel_x + 20, detail_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)
        bar_x = panel_x + 20
        bar_y = detail_y + 20
        bar_width = 300
        bar_height = 20
        cv2.rectangle(ui_image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                      (50, 50, 50), -1)
        fill_width = int(bar_width * intensity / 100)
        cv2.rectangle(ui_image, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height),
                      COLOR_BLUE, -1)

        detail_y += 60
        cv2.putText(ui_image, "Form Quality:", (panel_x + 20, detail_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)
        star_x = panel_x + 20
        star_y = detail_y + 25
        for i in range(5):
            if i < form_quality:
                cv2.circle(ui_image, (star_x + i * 30, star_y), 10, COLOR_BRIGHT_PURPLE, -1)
            else:
                cv2.circle(ui_image, (star_x + i * 30, star_y), 10, COLOR_PURPLE, 2)

        control_y = detail_y + 80
        button_width = 300
        button_height = 50
        button_spacing = 15

        btn1_y = control_y
        btn1_x = panel_x + 20
        button_coords['pause'] = (btn1_x, btn1_y, btn1_x + button_width, btn1_y + button_height)
        draw_rounded_rect(ui_image, (btn1_x, btn1_y),
                          (btn1_x + button_width, btn1_y + button_height),
                          COLOR_RED, -1, 10)
        cv2.putText(ui_image, "Pause" if not is_paused else "Resume",
                    (btn1_x + button_width // 2 - 40, btn1_y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_WHITE, 2)

        btn2_y = btn1_y + button_height + button_spacing
        btn2_x = panel_x + 20
        button_coords['change_exercise'] = (btn2_x, btn2_y, btn2_x + button_width, btn2_y + button_height)
        draw_rounded_rect(ui_image, (btn2_x, btn2_y),
                          (btn2_x + button_width, btn2_y + button_height),
                          COLOR_PURPLE, -1, 10)
        cv2.putText(ui_image, "Change Exercise",
                    (btn2_x + button_width // 2 - 80, btn2_y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)

        btn3_y = btn2_y + button_height + button_spacing
        btn3_x = panel_x + 20
        button_coords['end_workout'] = (btn3_x, btn3_y, btn3_x + button_width, btn3_y + button_height)
        draw_rounded_rect(ui_image, (btn3_x, btn3_y),
                          (btn3_x + button_width, btn3_y + button_height),
                          (40, 40, 40), -1, 10)
        cv2.putText(ui_image, "End Workout Session",
                    (btn3_x + button_width // 2 - 90, btn3_y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)

        time_y = btn3_y + button_height + 40
        cv2.putText(ui_image, "Workout Time", (panel_x + 20, time_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)
        time_text = format_time(workout_elapsed)
        cv2.putText(ui_image, time_text, (panel_x + 20, time_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_BRIGHT_PURPLE, 3)

        ui_image[video_y:video_y + VIDEO_HEIGHT, video_x:video_x + VIDEO_WIDTH] = video_image

        cv2.putText(ui_image, f"Prediction: {selected_exercise}",
                    (video_x + 10, video_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)

        # Display wrong rep alert (after video so it stays visible)
        if wrong_alert_start > 0 and current_time - wrong_alert_start < 3:
            alert_text = "Wrong Rep! Fix Your Form"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 3
            text_size = cv2.getTextSize(alert_text, font, font_scale, thickness)[0]
            text_x = video_x + VIDEO_WIDTH // 2 - text_size[0] // 2
            text_y = video_y + VIDEO_HEIGHT // 2 + text_size[1] // 2
            rect_start = (text_x - 24, video_y + VIDEO_HEIGHT // 2 - text_size[1] // 2 - 24)
            rect_end = (text_x + text_size[0] + 24, video_y + VIDEO_HEIGHT // 2 + text_size[1] // 2 + 24)

            overlay = ui_image.copy()
            cv2.rectangle(overlay, rect_start, rect_end, COLOR_RED, -1)
            cv2.addWeighted(overlay, 0.75, ui_image, 0.25, 0, ui_image)

            shadow_offset = 2
            cv2.putText(ui_image, alert_text,
                        (text_x + shadow_offset, text_y + shadow_offset),
                        font, font_scale, COLOR_BLACK, thickness + 1)
            cv2.putText(ui_image, alert_text,
                        (text_x, text_y),
                        font, font_scale, COLOR_WHITE, thickness)

        cv2.setMouseCallback('FitMaster AI - Bicep Curl Counter', mouse_callback)
        if mouse_clicked and not show_exercise_menu:
            mouse_clicked = False
            if button_coords['pause']:
                x1, y1, x2, y2 = button_coords['pause']
                if x1 <= mouse_x <= x2 and y1 <= mouse_y <= y2:
                    is_paused = not is_paused
                    if is_paused:
                        pause_start_time = current_time
                    else:
                        if pause_start_time > 0:
                            paused_time += (current_time - pause_start_time)
                            pause_start_time = 0

            if button_coords['change_exercise']:
                x1, y1, x2, y2 = button_coords['change_exercise']
                if x1 <= mouse_x <= x2 and y1 <= mouse_y <= y2:
                    show_exercise_menu = True
                    exercise_menu_start_time = current_time

            if button_coords['end_workout']:
                x1, y1, x2, y2 = button_coords['end_workout']
                if x1 <= mouse_x <= x2 and y1 <= mouse_y <= y2:
                    break

        if show_exercise_menu:
            menu_alpha = 0.9
            menu_width = 400
            menu_height = 300
            menu_x = WINDOW_WIDTH // 2 - menu_width // 2
            menu_y = WINDOW_HEIGHT // 2 - menu_height // 2

            overlay = ui_image.copy()
            cv2.rectangle(overlay, (menu_x, menu_y), (menu_x + menu_width, menu_y + menu_height),
                          COLOR_DARK_PURPLE, -1)
            cv2.addWeighted(overlay, menu_alpha, ui_image, 1 - menu_alpha, 0, ui_image)

            draw_rounded_rect(ui_image, (menu_x, menu_y), (menu_x + menu_width, menu_y + menu_height),
                              COLOR_BRIGHT_PURPLE, 3, 15)

            cv2.putText(ui_image, "Select Exercise", (menu_x + 100, menu_y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_WHITE, 2)

            exercise_buttons = []
            option_height = 40
            option_spacing = 10
            start_y = menu_y + 70

            for i, exercise in enumerate(available_exercises):
                option_y = start_y + i * (option_height + option_spacing)
                option_x = menu_x + 20
                option_width = menu_width - 40

                if exercise == selected_exercise:
                    draw_rounded_rect(ui_image, (option_x, option_y),
                                      (option_x + option_width, option_y + option_height),
                                      COLOR_PURPLE, -1, 8)
                else:
                    draw_rounded_rect(ui_image, (option_x, option_y),
                                      (option_x + option_width, option_y + option_height),
                                      COLOR_DARK_PURPLE, -1, 8)
                    draw_rounded_rect(ui_image, (option_x, option_y),
                                      (option_x + option_width, option_y + option_height),
                                      COLOR_PURPLE, 2, 8)

                exercise_display = exercise.replace('-', ' ').title()
                text_size = cv2.getTextSize(exercise_display, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                text_x = option_x + (option_width - text_size[0]) // 2
                cv2.putText(ui_image, exercise_display,
                            (text_x, option_y + 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)

                exercise_buttons.append((option_x, option_y, option_x + option_width, option_y + option_height, exercise))

            close_y = menu_y + menu_height - 50
            close_x = menu_x + menu_width - 120
            close_width = 100
            close_height = 35
            draw_rounded_rect(ui_image, (close_x, close_y),
                              (close_x + close_width, close_y + close_height),
                              COLOR_RED, -1, 8)
            cv2.putText(ui_image, "Close",
                        (close_x + 25, close_y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)

            if mouse_clicked:
                mouse_clicked = False
                for x1, y1, x2, y2, exercise in exercise_buttons:
                    if x1 <= mouse_x <= x2 and y1 <= mouse_y <= y2:
                        selected_exercise = exercise
                        show_exercise_menu = False
                        break

                if close_x <= mouse_x <= close_x + close_width and close_y <= mouse_y <= close_y + close_height:
                    show_exercise_menu = False

            if current_time - exercise_menu_start_time > 10:
                show_exercise_menu = False

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q') or key == ord('e'):
            break
        elif key == ord('p'):
            is_paused = not is_paused
            if is_paused:
                pause_start_time = current_time
            else:
                if pause_start_time > 0:
                    paused_time += (current_time - pause_start_time)
                    pause_start_time = 0
        elif key == ord('c'):
            show_exercise_menu = True
            exercise_menu_start_time = current_time

        cv2.imshow('FitMaster AI - Bicep Curl Counter', ui_image)
        out.write(ui_image)

cap.release()
out.release()
cv2.destroyAllWindows()
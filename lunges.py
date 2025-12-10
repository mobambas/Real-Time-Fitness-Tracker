import cv2
import mediapipe as mp
import numpy as np
import time
import os
import simpleaudio as sa
import random
from datetime import datetime, timedelta

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

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """Returns angle between 3 points (a, b, c) in degrees"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180 else 360 - angle

# UI Constants
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
VIDEO_WIDTH = 800
VIDEO_HEIGHT = 600
PANEL_WIDTH = 400

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
    """Add confetti particles at position"""
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
    """Update confetti particle positions"""
    global confetti_particles
    for particle in confetti_particles[:]:
        particle['x'] += particle['vx']
        particle['y'] += particle['vy']
        particle['vy'] += 0.2  # gravity
        particle['life'] -= 1
        if particle['life'] <= 0:
            confetti_particles.remove(particle)

def draw_confetti(image):
    """Draw confetti particles"""
    for particle in confetti_particles:
        cv2.circle(image, (int(particle['x']), int(particle['y'])), particle['size'], particle['color'], -1)

def draw_rounded_rect(image, pt1, pt2, color, thickness=-1, radius=10):
    """Draw a rounded rectangle"""
    x1, y1 = pt1
    x2, y2 = pt2
    
    if thickness == -1:
        # Filled rectangle
        cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(image, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(image, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(image, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(image, (x2 - radius, y2 - radius), radius, color, -1)
    else:
        # Outlined rectangle
        cv2.line(image, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(image, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(image, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(image, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        cv2.ellipse(image, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(image, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(image, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(image, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)

def format_time(seconds):
    """Format seconds to MM:SS"""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

def get_target_muscles(exercise):
    """Get target muscles for a given exercise"""
    muscle_map = {
        "squats": "Quadriceps, Glutes",
        "push-ups": "Chest, Triceps, Shoulders",
        "sit-ups": "Abs, Core",
        "lunges": "Quadriceps, Glutes, Hamstrings"
    }
    return muscle_map.get(exercise, "Multiple Muscle Groups")

# Lunge detection variables
counter_left = 0
counter_right = 0
incorrect_counter = 0
sets = 1
stage_left = "up"
stage_right = "up"
frame_threshold = 3
min_left_knee_angle = 180.0
min_right_knee_angle = 180.0
min_time_between_reps = 1.5
min_time_between_wrong_alerts = 2.0
last_rep_time_left = 0.0
last_rep_time_right = 0.0
last_wrong_alert_time = -2.0
left_down_frames = 0
right_down_frames = 0
left_up_frames = 0
right_up_frames = 0
wrong_alert_start = -1.0
workout_start_time = time.time()
is_paused = False
paused_time = 0
pause_start_time = 0
wrong_suppress_until = 0.0
last_audio_time = 0.0
back_bad_frames = 0
back_bad_flag = False
forward_leg_locked = None
forward_leg_lock_frames = 0
back_angle_threshold = 145  # allow slight lean
min_vis_threshold = 0.5     # require visibility to trust back check

# Button click handling
mouse_x = 0
mouse_y = 0
mouse_clicked = False
show_exercise_menu = False
exercise_menu_start_time = 0
selected_exercise = "lunges"

# Available exercises
available_exercises = ["squats", "push-ups", "sit-ups", "lunges"]

# Button coordinates (will be set in main loop)
button_coords = {
    'pause': None,
    'change_exercise': None,
    'end_workout': None
}

def mouse_callback(event, x, y, flags, param):
    """Handle mouse events"""
    global mouse_x, mouse_y, mouse_clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_x = x
        mouse_y = y
        mouse_clicked = True

cap = cv2.VideoCapture(0)

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
recording_name = f"lunges_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
out = cv2.VideoWriter(recording_name, fourcc, 20.0, (WINDOW_WIDTH, WINDOW_HEIGHT))

# Create window and set mouse callback
cv2.namedWindow('FitMaster AI - Lunge Counter')
# cv2.setMouseCallback('FitMaster AI - Lunge Counter', mouse_callback) # Moved to main loop

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Create UI canvas
        ui_image = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
        ui_image[:] = COLOR_BLACK

        # Resize video frame
        frame_resized = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
        video_image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        video_image.flags.writeable = False
        results = pose.process(video_image)
        video_image.flags.writeable = True
        video_image = cv2.cvtColor(video_image, cv2.COLOR_RGB2BGR)
        
        # Draw pose landmarks on video frame
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                video_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            )

        # Video will be placed after all drawing operations
        video_x = 20
        video_y = 120

        # Calculate workout time
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

        # Calculate total reps
        total_reps = counter_left + counter_right

        # Calculate form quality (based on correct vs incorrect reps)
        total_all_reps = total_reps + incorrect_counter
        if total_all_reps > 0:
            form_quality = max(1, min(5, int(5 * (total_reps / total_all_reps))))
        else:
            form_quality = 3

        # Calculate intensity (based on reps per minute)
        if workout_elapsed > 0:
            reps_per_min = (total_reps / workout_elapsed) * 60
            intensity = min(100, int(reps_per_min * 5))  # Scale to 0-100
        else:
            intensity = 0

        forward_leg = "None"
        left_knee_angle = 0.0
        right_knee_angle = 0.0

        if results.pose_landmarks:
            try:
                landmarks = results.pose_landmarks.landmark

                # Left leg keypoints (normalized coords)
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                left_hip_vis = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].visibility
                left_knee_vis = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility

                # Right leg keypoints
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                right_hip_vis = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility
                right_knee_vis = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility

                # Back angles per side
                left_back_angle = calculate_angle(left_shoulder := [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                                                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
                                                  left_hip, left_knee)
                right_back_angle = calculate_angle(right_shoulder := [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                                                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],
                                                   right_hip, right_knee)

                # Calculate angles
                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

                # Display live angles on video frame
                left_knee_x = int(left_knee[0] * VIDEO_WIDTH)
                left_knee_y = int(left_knee[1] * VIDEO_HEIGHT)
                right_knee_x = int(right_knee[0] * VIDEO_WIDTH)
                right_knee_y = int(right_knee[1] * VIDEO_HEIGHT)
                
                cv2.putText(video_image, f"{int(left_knee_angle)}°", 
                           (left_knee_x + 20, left_knee_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(video_image, f"{int(right_knee_angle)}°",
                           (right_knee_x + 20, right_knee_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # z and y for depth/height checks
                left_knee_z = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z
                right_knee_z = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z
                left_knee_y_coord = left_knee[1]
                right_knee_y_coord = right_knee[1]

                # Determine forward leg: prefer z if it seems meaningful, otherwise use y/x
                use_z = True
                if abs(left_knee_z - right_knee_z) < 0.02:
                    use_z = False  # z too similar/noisy, fallback to y-based check

                if use_z:
                    # MediaPipe z is negative toward camera; smaller (more negative) is closer
                    if left_knee_z < right_knee_z:
                        forward_leg = "Left"
                    elif right_knee_z < left_knee_z:
                        forward_leg = "Right"
                    else:
                        forward_leg = "None"
                else:
                    # fallback: use y (lower y = higher on image) logic
                    if left_knee_y_coord < right_knee_y_coord - 0.03:
                        forward_leg = "Left"
                    elif right_knee_y_coord < left_knee_y_coord - 0.03:
                        forward_leg = "Right"
                    else:
                        forward_leg = "None"

                # Stabilize forward leg to avoid flip-flopping
                if forward_leg == forward_leg_locked:
                    forward_leg_lock_frames = min(forward_leg_lock_frames + 1, 30)
                else:
                    forward_leg_lock_frames = 0
                if forward_leg_lock_frames >= 4:
                    pass
                else:
                    if forward_leg_locked is None and forward_leg != "None":
                        forward_leg_locked = forward_leg
                    elif forward_leg != "None":
                        forward_leg_locked = forward_leg
                forward_leg = forward_leg_locked or forward_leg

                # Back stability check (use forward side)
                active_back_angle = left_back_angle if forward_leg == "Left" else right_back_angle if forward_leg == "Right" else min(left_back_angle, right_back_angle)

                # Process left-forward case
                if not is_paused:
                    rep_event = False
                    if forward_leg == "Left":
                        # LEFT leg is forward: handle left leg states first
                        # Down detection
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
                            # back straightness while moving
                            if forward_leg == "Left" and left_hip_vis >= min_vis_threshold and left_knee_vis >= min_vis_threshold and min_left_knee_angle < 120:
                                if active_back_angle < back_angle_threshold:
                                    back_bad_frames += 1
                                else:
                                    back_bad_frames = 0
                                    back_bad_flag = False
                            if left_knee_angle > 160:
                                left_up_frames += 1
                                if left_up_frames >= frame_threshold:
                                    # complete rep
                                    if current_time - last_rep_time_left > min_time_between_reps:
                                        if min_left_knee_angle < 100:
                                            counter_left += 1
                                            last_rep_time_left = current_time
                                            wrong_suppress_until = current_time + 0.8
                                            if current_time - last_audio_time > 0.35:
                                                play_rep_audio(counter_left)
                                                last_audio_time = current_time
                                            # Add confetti on successful rep
                                            add_confetti(video_x + VIDEO_WIDTH // 2, video_y + VIDEO_HEIGHT // 2)
                                            rep_event = True
                                        elif 110 <= min_left_knee_angle < 150:
                                            if current_time - last_wrong_alert_time > min_time_between_wrong_alerts and current_time > wrong_suppress_until:
                                                wrong_alert_start = current_time
                                                incorrect_counter += 1
                                                last_wrong_alert_time = current_time
                                                wrong_suppress_until = current_time + 0.8
                                                if current_time - last_audio_time > 0.35:
                                                    play_wrong_audio()
                                                    last_audio_time = current_time
                                                rep_event = True
                                    stage_left = "up"
                                    min_left_knee_angle = 180.0
                                    left_up_frames = 0
                            else:
                                left_up_frames = 0

                        # make sure right counters don't falsely accumulate
                        right_down_frames = 0
                        right_up_frames = 0

                    # Process right-forward case
                    elif forward_leg == "Right":
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
                            if forward_leg == "Right" and right_hip_vis >= min_vis_threshold and right_knee_vis >= min_vis_threshold and min_right_knee_angle < 120:
                                if active_back_angle < back_angle_threshold:
                                    back_bad_frames += 1
                                else:
                                    back_bad_frames = 0
                                    back_bad_flag = False
                            if right_knee_angle > 160:
                                right_up_frames += 1
                                if right_up_frames >= frame_threshold:
                                    if current_time - last_rep_time_right > min_time_between_reps:
                                        if min_right_knee_angle < 100:
                                            counter_right += 1
                                            last_rep_time_right = current_time
                                            wrong_suppress_until = current_time + 0.8
                                            if current_time - last_audio_time > 0.35:
                                                play_rep_audio(counter_right)
                                                last_audio_time = current_time
                                            # Add confetti on successful rep
                                            add_confetti(video_x + VIDEO_WIDTH // 2, video_y + VIDEO_HEIGHT // 2)
                                            rep_event = True
                                        elif 110 <= min_right_knee_angle < 150:
                                            if current_time - last_wrong_alert_time > min_time_between_wrong_alerts and current_time > wrong_suppress_until:
                                                wrong_alert_start = current_time
                                                incorrect_counter += 1
                                                last_wrong_alert_time = current_time
                                                wrong_suppress_until = current_time + 0.8
                                                if current_time - last_audio_time > 0.35:
                                                    play_wrong_audio()
                                                    last_audio_time = current_time
                                                rep_event = True
                                    stage_right = "up"
                                    min_right_knee_angle = 180.0
                                    right_up_frames = 0
                            else:
                                right_up_frames = 0

                        # reset left frame counters to avoid accidental triggers
                        left_down_frames = 0
                        left_up_frames = 0

                    else:
                        # No clear forward leg: reset frame counters to avoid false triggers
                        left_down_frames = 0
                        right_down_frames = 0
                        left_up_frames = 0
                        right_up_frames = 0
                        # If knees are flexing without clear forward leg, mark incorrect once
                        if (left_knee_angle < 150 or right_knee_angle < 150) and current_time > wrong_suppress_until:
                            wrong_alert_start = current_time
                            incorrect_counter += 1
                            wrong_suppress_until = current_time + 0.8
                            if current_time - last_audio_time > 0.35:
                                play_wrong_audio()
                                last_audio_time = current_time

                # If back stays bad for several frames during movement, flag incorrect (only when visibility is good)
                if back_bad_frames >= (frame_threshold + 1) and not back_bad_flag and current_time > wrong_suppress_until and not rep_event:
                    wrong_alert_start = current_time
                    incorrect_counter += 1
                    back_bad_flag = True
                    wrong_suppress_until = current_time + 0.8
                    if current_time - last_audio_time > 0.35:
                        play_wrong_audio()
                        last_audio_time = current_time
                else:
                    if rep_event:
                        back_bad_frames = 0
                        back_bad_flag = False

                # (alert drawing moved after video placement)

            except Exception as e:
                print("Error inside landmarks block:", e)

        # Update and draw confetti
        update_confetti()
        draw_confetti(ui_image)

        # ========== TOP BAR ==========
        cv2.rectangle(ui_image, (0, 0), (WINDOW_WIDTH, 60), COLOR_DARK_PURPLE, -1)
        cv2.putText(ui_image, "FitMaster AI", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_BRIGHT_PURPLE, 2)
        
        # Profile button (right side)
        profile_x = WINDOW_WIDTH - 150
        draw_rounded_rect(ui_image, (profile_x, 15), (WINDOW_WIDTH - 20, 45), COLOR_PURPLE, -1, 15)
        cv2.putText(ui_image, "haro...", (profile_x + 10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)

        # ========== HEADER ==========
        header_y = 70
        cv2.putText(ui_image, "Real-time Pose Correction", (20, header_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR_BRIGHT_PURPLE, 3)
        
        # Instruction banner
        banner_y = header_y + 10
        draw_rounded_rect(ui_image, (20, banner_y + 5), (VIDEO_WIDTH + 20, banner_y + 40), COLOR_YELLOW, -1, 8)
        cv2.putText(ui_image, "Make sure to face sideways! Side view helps track your form more accurately.",
                   (30, banner_y + 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BLACK, 2)

        # ========== METRICS CARDS ==========
        card_y = banner_y + 50
        card_width = 160
        card_height = 80
        card_spacing = 15
        
        # Total Reps
        card1_x = 20
        total_all_reps = total_reps + incorrect_counter
        draw_rounded_rect(ui_image, (card1_x, card_y), (card1_x + card_width, card_y + card_height),
                         COLOR_DARK_PURPLE, -1, 10)
        cv2.putText(ui_image, "Total Reps", (card1_x + 10, card_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)
        cv2.putText(ui_image, str(total_all_reps), (card1_x + 10, card_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_BRIGHT_PURPLE, 3)

        # Correct Reps
        card2_x = card1_x + card_width + card_spacing
        draw_rounded_rect(ui_image, (card2_x, card_y), (card2_x + card_width, card_y + card_height),
                         COLOR_DARK_PURPLE, -1, 10)
        cv2.putText(ui_image, "Correct Reps", (card2_x + 10, card_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)
        cv2.putText(ui_image, str(total_reps), (card2_x + 10, card_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_GREEN, 3)

        # Incorrect Reps
        card3_x = card2_x + card_width + card_spacing
        draw_rounded_rect(ui_image, (card3_x, card_y), (card3_x + card_width, card_y + card_height),
                         COLOR_DARK_PURPLE, -1, 10)
        cv2.putText(ui_image, "Incorrect Reps", (card3_x + 10, card_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)
        cv2.putText(ui_image, str(incorrect_counter), (card3_x + 10, card_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_RED, 3)

        # Total Sets
        card4_x = card3_x + card_width + card_spacing
        draw_rounded_rect(ui_image, (card4_x, card_y), (card4_x + card_width, card_y + card_height),
                         COLOR_DARK_PURPLE, -1, 10)
        cv2.putText(ui_image, "Total Sets", (card4_x + 10, card_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)
        cv2.putText(ui_image, str(sets), (card4_x + 10, card_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_BRIGHT_PURPLE, 3)

        # Current Exercise
        card5_x = card4_x + card_width + card_spacing
        draw_rounded_rect(ui_image, (card5_x, card_y), (card5_x + card_width, card_y + card_height),
                         COLOR_DARK_PURPLE, -1, 10)
        cv2.putText(ui_image, "Current Exercise", (card5_x + 10, card_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
        exercise_display = selected_exercise.replace('-', ' ')
        cv2.putText(ui_image, exercise_display, (card5_x + 10, card_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_BRIGHT_PURPLE, 2)

        # Connection Status
        status_x = WINDOW_WIDTH - 345
        status_y = card_y + card_height + 10 # Position it below the last metric card
        cv2.putText(ui_image, "Connection Status: Connected", (status_x, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_GREEN, 2)
        
        # ========== RIGHT PANEL ==========
        panel_x = VIDEO_WIDTH + 40
        panel_y = card_y
        panel_height = WINDOW_HEIGHT - panel_y - 20

        # Panel background
        draw_rounded_rect(ui_image, (panel_x, panel_y), (WINDOW_WIDTH - 20, WINDOW_HEIGHT - 20),
                         COLOR_DARK_PURPLE, -1, 15)

        # Exercise Details
        detail_y = panel_y + 30
        cv2.putText(ui_image, "Exercise Details", (panel_x + 20, detail_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_WHITE, 2)

        # Target Muscles
        detail_y += 40
        cv2.putText(ui_image, "Target Muscles:", (panel_x + 20, detail_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)
        target_muscles = get_target_muscles(selected_exercise)
        cv2.putText(ui_image, target_muscles, (panel_x + 20, detail_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BRIGHT_PURPLE, 2)

        # Rep Breakdown
        detail_y += 60
        total_all_reps = total_reps + incorrect_counter
        cv2.putText(ui_image, "Rep Breakdown:", (panel_x + 20, detail_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)
        detail_y += 25
        cv2.putText(ui_image, f"Total: {total_all_reps}", (panel_x + 20, detail_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)
        detail_y += 20
        cv2.putText(ui_image, f"Correct: {total_reps}", (panel_x + 20, detail_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_GREEN, 2)
        detail_y += 20
        cv2.putText(ui_image, f"Incorrect: {incorrect_counter}", (panel_x + 20, detail_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_RED, 2)
        if total_all_reps > 0:
            accuracy = int((total_reps / total_all_reps) * 100)
            detail_y += 20
            cv2.putText(ui_image, f"Accuracy: {accuracy}%", (panel_x + 20, detail_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_YELLOW, 2)

        # Left/Right Breakdown (compact to save vertical space)
        detail_y += 60
        cv2.putText(ui_image, "Left:", (panel_x + 20, detail_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)
        cv2.putText(ui_image, str(counter_left), (panel_x + 90, detail_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_BLUE, 2)

        cv2.putText(ui_image, "Right:", (panel_x + 150, detail_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)
        cv2.putText(ui_image, str(counter_right), (panel_x + 230, detail_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_GREEN, 2)

        # Intensity
        detail_y += 50
        cv2.putText(ui_image, "Intensity:", (panel_x + 20, detail_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)
        # Progress bar background
        bar_x = panel_x + 20
        bar_y = detail_y + 20
        bar_width = 300
        bar_height = 20
        cv2.rectangle(ui_image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     (50, 50, 50), -1)
        # Progress bar fill
        fill_width = int(bar_width * intensity / 100)
        cv2.rectangle(ui_image, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height),
                     COLOR_BLUE, -1)

        # Form Quality
        detail_y += 60
        cv2.putText(ui_image, "Form Quality:", (panel_x + 20, detail_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)
        # Stars
        star_x = panel_x + 20
        star_y = detail_y + 25
        for i in range(5):
            if i < form_quality:
                # Filled star (simplified as circle)
                cv2.circle(ui_image, (star_x + i * 30, star_y), 10, COLOR_BRIGHT_PURPLE, -1)
            else:
                # Empty star (outline)
                cv2.circle(ui_image, (star_x + i * 30, star_y), 10, COLOR_PURPLE, 2)

        # Controls
        control_y = detail_y + 60
        button_width = 300
        button_height = 50
        button_spacing = 10

        # Pause button
        btn1_y = control_y
        btn1_x = panel_x + 20
        button_coords['pause'] = (btn1_x, btn1_y, btn1_x + button_width, btn1_y + button_height)
        draw_rounded_rect(ui_image, (btn1_x, btn1_y),
                         (btn1_x + button_width, btn1_y + button_height),
                         COLOR_RED, -1, 10)
        cv2.putText(ui_image, "Pause" if not is_paused else "Resume",
                   (btn1_x + button_width // 2 - 40, btn1_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_WHITE, 2)

        # Change Exercise button
        btn2_y = btn1_y + button_height + button_spacing
        btn2_x = panel_x + 20
        button_coords['change_exercise'] = (btn2_x, btn2_y, btn2_x + button_width, btn2_y + button_height)
        draw_rounded_rect(ui_image, (btn2_x, btn2_y),
                         (btn2_x + button_width, btn2_y + button_height),
                         COLOR_PURPLE, -1, 10)
        cv2.putText(ui_image, "Change Exercise",
                   (btn2_x + button_width // 2 - 80, btn2_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)

        # End Workout button
        btn3_y = btn2_y + button_height + button_spacing
        btn3_x = panel_x + 20
        button_coords['end_workout'] = (btn3_x, btn3_y, btn3_x + button_width, btn3_y + button_height)
        draw_rounded_rect(ui_image, (btn3_x, btn3_y),
                         (btn3_x + button_width, btn3_y + button_height),
                         (40, 40, 40), -1, 10)
        cv2.putText(ui_image, "End Workout Session",
                   (btn3_x + button_width // 2 - 90, btn3_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)

        # Workout Time
        time_y = btn3_y + button_height + 40
        cv2.putText(ui_image, "Workout Time", (panel_x + 20, time_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)
        time_text = format_time(workout_elapsed)
        cv2.putText(ui_image, time_text, (panel_x + 20, time_y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_BRIGHT_PURPLE, 3)

        # Place video on UI (after all drawing operations)
        ui_image[video_y:video_y+VIDEO_HEIGHT, video_x:video_x+VIDEO_WIDTH] = video_image
        
        # Display prediction on video
        cv2.putText(ui_image, f"Prediction: {selected_exercise}",
                   (video_x + 10, video_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)
        
        # Display forward leg info (compact, top-right of video to save space)
        cv2.putText(ui_image, f"Forward: {forward_leg}",
                   (video_x + VIDEO_WIDTH - 200, video_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_YELLOW, 2)

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

        # === BUTTON COORDS MUST BE DEFINED BEFORE CLICK HANDLING ===
        # ... button_coords assignment for all buttons ...

        # Handle button clicks (only if menu is not showing)
        cv2.setMouseCallback('FitMaster AI - Lunge Counter', mouse_callback)
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
            
            # Check Change Exercise button
            if button_coords['change_exercise']:
                x1, y1, x2, y2 = button_coords['change_exercise']
                if x1 <= mouse_x <= x2 and y1 <= mouse_y <= y2:
                    show_exercise_menu = True
                    exercise_menu_start_time = current_time
            
            # Check End Workout button
            if button_coords['end_workout']:
                x1, y1, x2, y2 = button_coords['end_workout']
                if x1 <= mouse_x <= x2 and y1 <= mouse_y <= y2:
                    break

        # Draw exercise selection menu
        if show_exercise_menu:
            menu_alpha = 0.9
            menu_width = 400
            menu_height = 300
            menu_x = WINDOW_WIDTH // 2 - menu_width // 2
            menu_y = WINDOW_HEIGHT // 2 - menu_height // 2
            
            # Semi-transparent overlay
            overlay = ui_image.copy()
            cv2.rectangle(overlay, (menu_x, menu_y), (menu_x + menu_width, menu_y + menu_height),
                         COLOR_DARK_PURPLE, -1)
            cv2.addWeighted(overlay, menu_alpha, ui_image, 1 - menu_alpha, 0, ui_image)
            
            # Menu border
            draw_rounded_rect(ui_image, (menu_x, menu_y), (menu_x + menu_width, menu_y + menu_height),
                             COLOR_BRIGHT_PURPLE, 3, 15)
            
            # Menu title
            cv2.putText(ui_image, "Select Exercise", (menu_x + 100, menu_y + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_WHITE, 2)
            
            # Exercise options
            exercise_buttons = []
            option_height = 40
            option_spacing = 10
            start_y = menu_y + 70
            
            for i, exercise in enumerate(available_exercises):
                option_y = start_y + i * (option_height + option_spacing)
                option_x = menu_x + 20
                option_width = menu_width - 40
                
                # Highlight selected exercise
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
                
                # Exercise name (capitalize)
                exercise_display = exercise.replace('-', ' ').title()
                text_size = cv2.getTextSize(exercise_display, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                text_x = option_x + (option_width - text_size[0]) // 2
                cv2.putText(ui_image, exercise_display,
                           (text_x, option_y + 28),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)
                
                exercise_buttons.append((option_x, option_y, option_x + option_width, option_y + option_height, exercise))
            
            # Close button
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
            
            # Check for clicks on exercise menu
            if mouse_clicked:
                mouse_clicked = False
                # Check exercise options
                for x1, y1, x2, y2, exercise in exercise_buttons:
                    if x1 <= mouse_x <= x2 and y1 <= mouse_y <= y2:
                        selected_exercise = exercise
                        show_exercise_menu = False
                        # Update current exercise display
                        break
                
                # Check close button
                if close_x <= mouse_x <= close_x + close_width and close_y <= mouse_y <= close_y + close_height:
                    show_exercise_menu = False
            
            # Auto-close menu after 10 seconds
            if current_time - exercise_menu_start_time > 10:
                show_exercise_menu = False

        # Keyboard controls
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
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
        elif key == ord('e'):
            break

        # Show and save
        cv2.imshow('FitMaster AI - Lunge Counter', ui_image)
        out.write(ui_image)

cap.release()
out.release()
cv2.destroyAllWindows()
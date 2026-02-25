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

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

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
        "lunges": "Quadriceps, Glutes, Hamstrings",
        "bicep curls": "Biceps, Forearms",
        "lat pulldown": "Lats, Biceps, Rear Delts",
        "pull-ups": "Lats, Biceps, Core"
    }
    return muscle_map.get(exercise, "Multiple Muscle Groups")

# Repetition counter variables
counter = 0
incorrect_counter = 0
sets = 1
stage = "up"  # start at top position
down_frames = 0
up_frames = 0
partial_down_frames = 0
frame_threshold = 3  # buffer to confirm position and reduce noise
depth_reached = False  # true when a full bottom position is achieved
back_bad_frames = 0
back_bad_flag = False  # prevents double-counting back errors

# Time buffer to avoid double-counting
last_rep_time = 0
min_time_between_reps = 0.25  # seconds

# Wrong rep alert variables
wrong_alert_start = -1
partial_rep_detected = False  # Track if a partial rep was started

# Workout tracking
workout_start_time = time.time()
is_paused = False
paused_time = 0
pause_start_time = 0

# Consecutive counters and popup state
consecutive_incorrect = 0
consecutive_correct = 0
instruction_popup_active = False
instruction_text = ""
success_popup_start = None
SUCCESS_POPUP_DURATION = 3.0
# Per-rep flag: when True this rep was already marked incorrect
current_rep_incorrect = False

def register_correct_rep(n: int = 1):
    global counter, consecutive_correct, consecutive_incorrect, success_popup_start, current_rep_incorrect
    # If this rep was already marked incorrect earlier, skip counting it as correct
    if current_rep_incorrect:
        current_rep_incorrect = False
        consecutive_correct = 0
        return
    counter += n
    consecutive_correct += 1
    consecutive_incorrect = 0
    if consecutive_correct >= 5:
        success_popup_start = time.time()

def register_incorrect_rep(n: int = 1):
    global incorrect_counter, consecutive_incorrect, consecutive_correct, instruction_popup_active, instruction_text, current_rep_incorrect
    incorrect_counter += n
    consecutive_incorrect += 1
    consecutive_correct = 0
    # mark this rep incorrect so a later correct-detection won't count
    current_rep_incorrect = True
    if consecutive_incorrect >= 3:
        instruction_popup_active = True
        instruction_text = (
            "Proper Push-Up Form:\n"
            "- Keep a straight line from head to heels; avoid sagging hips.\n"
            "- Elbow angle should come below ~95 degrees at the bottom.\n"
            "- Maintain tight core and controlled tempo.\n"
            "- Avoid flaring elbows excessively.\n\n"
            "Learn more:\n"
            "https://www.exrx.net/WeightExercises/Pectorals/BWPushup"
        )

# Button click handling
mouse_x = 0
mouse_y = 0
mouse_clicked = False
show_exercise_menu = False
exercise_menu_start_time = 0
selected_exercise = "push-ups"

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

# Video capture
cap = cv2.VideoCapture(0)

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
recording_name = f"pushup_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
out = cv2.VideoWriter(recording_name, fourcc, 20.0, (WINDOW_WIDTH, WINDOW_HEIGHT))

# Create window and set mouse callback
cv2.namedWindow('FitMaster AI - Push-Up Counter')
cv2.setMouseCallback('FitMaster AI - Push-Up Counter', mouse_callback)

# Pose detection
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

        # Calculate form quality (based on correct vs incorrect reps)
        total_reps = counter + incorrect_counter
        if total_reps > 0:
            form_quality = max(1, min(5, int(5 * (counter / total_reps))))
        else:
            form_quality = 3

        # Calculate intensity (based on reps per minute)
        if workout_elapsed > 0:
            reps_per_min = (counter / workout_elapsed) * 60
            intensity = min(100, int(reps_per_min * 5))  # Scale to 0-100
        else:
            intensity = 0

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

            # Display live angles on video frame
            elbow_x = int(elbow_coords[0] * VIDEO_WIDTH)
            elbow_y = int(elbow_coords[1] * VIDEO_HEIGHT)
            
            cv2.putText(video_image, f"{int(angle)}°", 
                       (elbow_x + 20, elbow_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(video_image, f"Back: {int(back_angle)}°",
                       (elbow_x + 20, elbow_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Push-up detection: require full cycle (up -> down -> up) for a rep
            if not is_paused:
                # Track back straightness only when moving meaningfully
                if back_angle < 145 and (angle < 150 or stage == "down" or depth_reached or partial_rep_detected):
                    back_bad_frames += 1
                else:
                    back_bad_frames = 0
                    back_bad_flag = False

                # At top (arms extended)
                if angle > 160 and back_angle > 160:
                    up_frames += 1
                    down_frames = 0
                    partial_down_frames = 0
                    if up_frames >= frame_threshold:
                        # Complete rep only if we previously hit depth
                        if stage == "down" and depth_reached and (current_time - last_rep_time > min_time_between_reps):
                                register_correct_rep()
                                last_rep_time = current_time
                                play_rep_audio(counter)
                                add_confetti(video_x + VIDEO_WIDTH // 2, video_y + VIDEO_HEIGHT // 2)
                        # If we never hit depth but moved back up, count an incorrect rep
                        elif partial_rep_detected and not depth_reached and (current_time - last_rep_time > min_time_between_reps):
                            wrong_alert_start = current_time
                            register_incorrect_rep()
                            play_wrong_audio()
                        stage = "up"
                        depth_reached = False
                        partial_rep_detected = False

                # At bottom (sufficient depth, straight back)
                elif angle < 95 and back_angle > 160:
                    down_frames += 1
                    up_frames = 0
                    partial_down_frames = 0
                    partial_rep_detected = False
                    if down_frames >= frame_threshold:
                        depth_reached = True
                        stage = "down"

                # Partial depth while descending
                elif 95 <= angle <= 130 and back_angle > 160 and stage == "up":
                    partial_down_frames += 1
                    down_frames = 0
                    up_frames = 0
                    if partial_down_frames >= frame_threshold:
                        partial_rep_detected = True

                # Back not straight during the movement -> incorrect rep
                if back_bad_frames >= frame_threshold and not back_bad_flag and angle < 150:
                    wrong_alert_start = current_time
                    register_incorrect_rep()
                    back_bad_flag = True
                    play_wrong_audio()

        except:
            pass

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
        total_reps = counter + incorrect_counter
        draw_rounded_rect(ui_image, (card1_x, card_y), (card1_x + card_width, card_y + card_height),
                         COLOR_DARK_PURPLE, -1, 10)
        cv2.putText(ui_image, "Total Reps", (card1_x + 10, card_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)
        cv2.putText(ui_image, str(total_reps), (card1_x + 10, card_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_BRIGHT_PURPLE, 3)

        # Correct Reps
        card2_x = card1_x + card_width + card_spacing
        draw_rounded_rect(ui_image, (card2_x, card_y), (card2_x + card_width, card_y + card_height),
                         COLOR_DARK_PURPLE, -1, 10)
        cv2.putText(ui_image, "Correct Reps", (card2_x + 10, card_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)
        cv2.putText(ui_image, str(counter), (card2_x + 10, card_y + 60),
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
        status_y = card_y + 15
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
        total_reps = counter + incorrect_counter
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

        # Intensity
        detail_y += 60
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
        control_y = detail_y + 80
        button_width = 300
        button_height = 50
        button_spacing = 15

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

        # === BUTTON COORDS MUST BE DEFINED BEFORE CLICK HANDLING ===
        # ... button_coords assignment code here as in the main UI loop ...

        # Handle button clicks (only if menu is not showing)
        cv2.setMouseCallback('FitMaster AI - Push-Up Counter', mouse_callback)
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

        # Display wrong rep alert (drawn after video so it stays visible)
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

        # Persistent instruction popup
        def draw_persistent_instruction(img, text):
            h, w = img.shape[:2]
            box_w = int(w * 0.5)
            box_h = int(h * 0.45)
            x = (w - box_w) // 2
            y = (h - box_h) // 2
            overlay = img.copy()
            cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (50, 50, 50), -1)
            cv2.addWeighted(overlay, 0.9, img, 0.1, 0, img)
            cv2.putText(img, "Instruction", (x + 12, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            lines = text.split('\n')
            ty = y + 60
            for line in lines:
                cv2.putText(img, line, (x + 14, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)
                ty += 26

        # Success popup (auto-dismisses)
        def draw_success_popup(img, start_time):
            global success_popup_start
            if start_time is None:
                return
            elapsed = time.time() - start_time
            if elapsed > SUCCESS_POPUP_DURATION:
                success_popup_start = None
                return
            h, w = img.shape[:2]
            box_w = 360
            box_h = 80
            x = (w - box_w) // 2
            y = 20
            cv2.rectangle(img, (x, y), (x + box_w, y + box_h), (0, 160, 0), -1)
            cv2.putText(img, "Good job! Keep it up.", (x + 20, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        if instruction_popup_active:
            h, w = ui_image.shape[:2]
            box_w = int(w * 0.5)
            box_h = int(h * 0.45)
            x = (w - box_w) // 2
            y = (h - box_h) // 2
            draw_persistent_instruction(ui_image, instruction_text)
            close_rect = (x + box_w - 90, y + 6, x + box_w - 18, y + 28)
            if mouse_clicked:
                mx, my = mouse_x, mouse_y
                if close_rect[0] <= mx <= close_rect[2] and close_rect[1] <= my <= close_rect[3]:
                    instruction_popup_active = False
                    consecutive_incorrect = 0
                mouse_clicked = False

        if success_popup_start is not None:
            draw_success_popup(ui_image, success_popup_start)


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

        # Keyboard: dismiss instruction popup with 'i'
        if key == ord('i'):
            instruction_popup_active = False
            consecutive_incorrect = 0

        # Show and save
        cv2.imshow('FitMaster AI - Push-Up Counter', ui_image)
        out.write(ui_image)

cap.release()
out.release()
cv2.destroyAllWindows()
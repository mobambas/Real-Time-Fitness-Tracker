## Mobile Architecture (Flutter / Native)

This document describes how to turn the existing Python CV prototype into a real-time mobile app. The examples assume **Flutter**, but the same ideas apply to native Kotlin/Swift.

### High-Level Flow

- **Camera**: Use `camera` or `camera_platform_interface` (Flutter) or CameraX (Android) to stream frames.
- **Pose Model**: Run **MediaPipe Pose / BlazePose** or a TFLite pose model on-device to get 2D (and optionally 3D) landmarks.
- **Exercise Logic**: Port the Python angle/rep logic into Dart/Kotlin, operating only on landmark coordinates and timestamps.
- **UI**: Render camera preview + overlays (rep counts, alerts, streaks) at 30–60 FPS.
- **Backend Sync**: At the end of a session, POST summary metrics to the Vercel/Supabase backend.

The heavy work (pose inference and angle calculations) stays on-device for low latency and privacy.

### Data Model for Landmarks

Use a minimal representation that matches MediaPipe’s output:

```dart
class Landmark {
  final double x; // normalized [0,1]
  final double y; // normalized [0,1]
  final double z; // normalized depth
  final double visibility; // [0,1]

  const Landmark({required this.x, required this.y, required this.z, required this.visibility});
}

typedef PoseFrame = Map<String, Landmark>; // e.g. "LEFT_SHOULDER", "RIGHT_KNEE"
```

### Porting the Angle Utility

The Python scripts use the same helper:

- `bicep_curls.py`, `squats.py`, `push-up.py`, `lunges.py`, `pull_ups.py`, `lat_pulldown.py`, `sit-up.py` all call a `calculate_angle(a, b, c)` with three 2D points.

Port it directly:

```dart
import 'dart:math' as math;

double calculateAngle(List<double> a, List<double> b, List<double> c) {
  final ax = a[0], ay = a[1];
  final bx = b[0], by = b[1];
  final cx = c[0], cy = c[1];

  final ab = math.atan2(ay - by, ax - bx);
  final cb = math.atan2(cy - by, cx - bx);
  var angle = (cb - ab) * 180.0 / math.pi;
  angle = angle.abs();
  return angle <= 180 ? angle : 360 - angle;
}
```

### Example: Push-Up Logic in Dart

The Python `push-up.py` roughly:

- Chooses the arm closer to the camera.
- Computes elbow angle (shoulder–elbow–wrist) and back angle (shoulder–hip–knee).
- Counts a rep when you go from **top** (`angle > 160`, `back_angle > 160`) → **bottom** (`angle < ~95`) → back to **top**.
- Flags incorrect reps if the back angle drops (hips sagging, poor plank).

You can implement an equivalent state machine in Dart:

```dart
class PushUpState {
  int correctCount = 0;
  int incorrectCount = 0;
  String stage = 'up'; // 'up' | 'down'
  int upFrames = 0;
  int downFrames = 0;
  int backBadFrames = 0;
  bool backBadFlag = false;
  bool partialRep = false;
  double lastRepTime = 0;
}

class PushUpUpdateResult {
  final int correct;
  final int incorrect;
  final bool justCountedCorrect;
  final bool justCountedIncorrect;
  PushUpUpdateResult({
    required this.correct,
    required this.incorrect,
    required this.justCountedCorrect,
    required this.justCountedIncorrect,
  });
}

PushUpUpdateResult updatePushUp(
  PushUpState s,
  PoseFrame pose,
  double t, // seconds
  int frameThreshold,
) {
  final ls = pose['LEFT_SHOULDER']!, le = pose['LEFT_ELBOW']!, lw = pose['LEFT_WRIST']!;
  final rs = pose['RIGHT_SHOULDER']!, re = pose['RIGHT_ELBOW']!, rw = pose['RIGHT_WRIST']!;
  final lh = pose['LEFT_HIP']!, lk = pose['LEFT_KNEE']!;
  final rh = pose['RIGHT_HIP']!, rk = pose['RIGHT_KNEE']!;

  final useLeft = le.z < re.z;
  final shoulder = useLeft ? ls : rs;
  final elbow = useLeft ? le : re;
  final wrist = useLeft ? lw : rw;
  final hip = useLeft ? lh : rh;
  final knee = useLeft ? lk : rk;

  final elbowAngle = calculateAngle(
    [shoulder.x, shoulder.y],
    [elbow.x, elbow.y],
    [wrist.x, wrist.y],
  );
  final backAngle = calculateAngle(
    [shoulder.x, shoulder.y],
    [hip.x, hip.y],
    [knee.x, knee.y],
  );

  bool justCorrect = false;
  bool justIncorrect = false;

  // Back-straight check during movement
  if (backAngle < 145 && elbowAngle < 150) {
    s.backBadFrames++;
  } else {
    s.backBadFrames = 0;
    s.backBadFlag = false;
  }
  if (s.backBadFrames >= frameThreshold && !s.backBadFlag) {
    s.incorrectCount++;
    s.backBadFlag = true;
    justIncorrect = true;
  }

  // Top position
  if (elbowAngle > 160 && backAngle > 160) {
    s.upFrames++;
    s.downFrames = 0;
    if (s.upFrames >= frameThreshold) {
      if (s.stage == 'down' && s.partialRep == false && t - s.lastRepTime > 0.25) {
        s.correctCount++;
        s.lastRepTime = t;
        justCorrect = true;
      } else if (s.partialRep && t - s.lastRepTime > 0.25) {
        s.incorrectCount++;
        s.lastRepTime = t;
        justIncorrect = true;
      }
      s.stage = 'up';
      s.partialRep = false;
    }
  }
  // Bottom position (full depth)
  else if (elbowAngle < 95 && backAngle > 160) {
    s.downFrames++;
    s.upFrames = 0;
    if (s.downFrames >= frameThreshold) {
      s.stage = 'down';
      s.partialRep = false;
    }
  }
  // Partial
  else if (elbowAngle >= 95 && elbowAngle <= 130 && backAngle > 160 && s.stage == 'up') {
    s.partialRep = true;
  }

  return PushUpUpdateResult(
    correct: s.correctCount,
    incorrect: s.incorrectCount,
    justCountedCorrect: justCorrect,
    justCountedIncorrect: justIncorrect,
  );
}
```

The same pattern can be applied to:

- **Squats**: use hip–knee–ankle angles, forward hip drop, and back angle (see `squats.py`).
- **Bicep curls / Lat pulldown**: use shoulder–elbow–wrist angle and optional back angle, as in `bicep_curls.py` and `lat_pulldown.py`.
- **Lunges**: use per-leg knee angles with thresholds (around 90–120°) and optional back checks, as in `lunges.py`.
- **Pull-ups**: use shoulder/elbow angles and vertical displacement relative to a calibrated “bar” height, as in `pull_ups.py`.
- **Sit-ups**: use hip angle and nose/shoulder Y distance from a baseline down position, as in `sit-up.py`.

Keep thresholds as constants so you can fine-tune per device:

- Push-up bottom: elbow angle \(< 95^\circ\), back angle \(> 160^\circ\).
- Squat depth: knee angle \(< 115\text{–}120^\circ\).
- Bicep curl peak: elbow angle \(< 50^\circ\).
- Lunge depth: front knee \( \approx 90^\circ \pm 10^\circ \).

### Performance Considerations

- **Drop frames**: only run full pose inference on every 2nd–3rd frame, or use a “keep latest only” camera strategy.
- **Resize input**: feed the model \(224 \times 224\) or similar while rendering full-resolution preview.
- **Threads**: run inference and exercise logic on a background isolate/thread; keep UI thread for rendering only.
- **GPU / NNAPI / CoreML**: enable hardware acceleration where possible (TFLite GPU delegate or MediaPipe’s GPU path).

### Session Metrics for Backend

On mobile, at the end of a workout send a JSON payload to the backend:

```json
{
  "user_id": "clerk_user_id",
  "exercise_type": "push-ups",
  "started_at": "2025-12-10T14:03:00Z",
  "ended_at": "2025-12-10T14:08:00Z",
  "correct_reps": 42,
  "incorrect_reps": 5,
  "duration_seconds": 300,
  "device": "android",
  "metrics": {
    "avg_form_score": 4.2,
    "max_intensity": 85
  }
}
```

The Supabase schema in `docs/BACKEND_SUPABASE_SCHEMA.sql` and the Vercel API examples in `backend/` expect a payload similar to this.


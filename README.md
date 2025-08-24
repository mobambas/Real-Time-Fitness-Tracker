# 🏋️‍♂️ Real-Time Fitness Tracker: Crush Your Workouts with AI!  

Get pumped with the **Real-Time Fitness Tracker**—an AI-powered app built with **Python**, **MediaPipe**, and **OpenCV** that tracks your **push-ups, sit-ups, squats, and lunges** with precision.  

This isn’t just about counting reps—it’s about **perfect form**, **real-time feedback**, **audio hype**, and even a **fitness chatbot** powered by Google Gemini AI. 🚀  

---

## 🌟 Features at a Glance  

- **⚡ Real-Time Pose Tracking**: MediaPipe PoseLandmark keeps your movements sharp and accurate.  
- **📊 Rep Counting with Logic**:  
  - 🥊 **Push-ups**: Counts reps when elbows < **100°** and back > **160°**.  
  - 🧘 **Sit-ups**: Tracks with hip angles & shoulder/nose movement.  
  - 🏋️ **Squats**: Logs reps when knees < **120°**.  
  - 🦵 **Lunges**: Tracks each leg separately when knee < **100°**.  
- **🛡️ Form Feedback**: Half-reps trigger **“Go Deeper!”** alerts (with audio).  
- **🔊 Audio Feedback**: Hear reps counted aloud (`1`, `2`, `3`...) and a cheeky **wrong.wav** if form slips.  
- **🎥 Workout Replay**: Save sessions as MP4 (`pushup_session.mp4`, `squat_session.mp4`, etc.).  
- **📈 Live Stats**: Angles, rep counts, and exercise stage displayed in real time.  
- **🤖 Fitness Chatbot**: Ask workout tips anytime—powered by **Google Gemini AI**.  
- **🎯 Robust Design**: Works with multiple camera angles and body positions.  

---

## 🛠️ Tech Stack  

- **Python 3.8+** – Core language  
- **OpenCV** – Video processing & overlay  
- **MediaPipe** – Pose estimation  
- **NumPy** – Angle calculations  
- **SimpleAudio** – Rep-counting & alerts  
- **Google Gemini AI** – Chatbot intelligence  
- **python-dotenv** – Secure API key storage  

---

### Install Dependencies
Fire up your terminal and install the required libraries:
```bash
pip install opencv-python mediapipe numpy simpleaudio google-genai python-dotenv
```
---

## 🏃‍♂️ Setup: Let’s Get Sweaty!

1. **Grab the Code**:
   ```bash
   git clone https://github.com/mobambas/Real-Time-Fitness-Tracker.git
   cd Real-Time-Fitness-Tracker
   ```

2. **Set Up Audio Vibes**:
   - Drop audio files (`1.wav`, `2.wav`, ..., `wrong.wav`) into an `audio/` folder in the project root.
   - No files? Record your own or use a text-to-speech tool for numbered rep sounds and a “wrong.wav” alert.

3. **Configure the Chatbot**:
   - Create a `.env` file in the project root with your Google API key:
     ```
     GOOGLE_API_KEY=your-api-key-here
     ```
   - Get your API key from [Google Cloud Console](https://console.cloud.google.com/) by enabling the Gemini API.

4. **Check Your Camera**:
   - Scripts use `cv2.VideoCapture(0)` for internal webcam. For an external camera, switch to `1` or test with `cv2.VideoCapture(1)`.

5. **Install the Good Stuff**:
   Create a `requirements.txt` file with:
   ```
   opencv-python
   mediapipe
   numpy
   simpleaudio
   google-genai
   python-dotenv
   ```
   Then run:
   ```bash
   pip install -r requirements.txt
   ```
---

## 💪 How to Use It

1. **Launch an Exercise**:
   Pick your workout and run the script:
   ```bash
   python Push-ups.py
   ```
   Choose from:
   - `Push-ups.py`: Crush those push-ups.
   - `Sit-ups.py`: Power through sit-ups.
   - `Squats.py`: Drop it low with squats.
   - `Lunges.py`: Balance it out with lunges (tracks both legs!).

2. **Work It**:
   - Stand in front of your camera, and the app roars to life!
   - Watch real-time joint angles, rep counts, and form feedback on-screen.
   - Hit `q` to wrap up your session.
   - Your workout is saved as an MP4 for epic replays.

3. **Chat with Your Fitness Guru**:
   - Run the chatbot:
     ```bash
     python chatbot.py
     ```
   - Ask about exercise form, workout plans, or recovery tips (e.g., “How do I improve my push-up form?”).
   - Type `exit` to quit the chatbot.
   - The chatbot uses Google’s Gemini AI to deliver concise, fitness-focused answers (e.g., “1. Keep elbows at 45°. 2. Maintain a straight back.”).

4. **What You Get**:
   - MP4 videos (`pushup_session.mp4`, `situp_session.mp4`, etc.) to review your sessions.
   - Real-time feedback with rep counts, angles, and sassy “Go Deeper!” alerts.
   - Expert fitness advice from the chatbot, anytime!

---

## 🧠 How It Works: The Tech Magic

- **Pose Detection**: MediaPipe scans video frames to pinpoint body landmarks (shoulders, elbows, hips, knees).
- **Angle Wizardry**: NumPy calculates joint angles for spot-on rep detection.
- **Rep Counting Logic**:
  - **Push-ups**: Counts when elbows bend <100° with a straight back (>160°).
  - **Sit-ups**: Tracks shoulder/nose shifts and hip angles (<170°) for clean reps.
  - **Squats**: Logs reps when knees dip below 120°.
  - **Lunges**: Monitors both legs’ knee angles (<100°) for balanced tracking.
- **Form Police**: Partial reps trigger audio and visual “Go Deeper!” alerts to keep you honest.
- **Video Output**: OpenCV saves every frame with stats overlaid for post-workout glory.
- **Chatbot Brains**: Google Gemini AI powers the chatbot, cleaning responses for clear, numbered lists of fitness tips.

---

## 📂 Project Structure

```
├── audio/
│   ├── 1.wav
│   ├── 2.wav
│   ├── ...
│   ├── wrong.wav
├── .env
├── chatbot.py
├── push-ups.py
├── sit-ups.py
├── squats.py
├── lunges.py
├── requirements.txt
├── LICENSE
├── README.md
```
---

## 🔥 What’s Next? Let’s Go Big!

- Add more exercises like planks or yoga poses for ultimate workout domination.
- Build a slick GUI to pick exercises and chat with the bot.
- Supercharge the chatbot with advanced NLP for even smarter fitness advice.
- Track rep speed and form consistency for pro-level analytics.
- Save workout videos to the cloud for easy access anywhere.

---

## 🤝 Join the Fitness Revolution

Want to make this project even more epic? Here’s how to contribute:
1. Fork the repo.
2. Create a branch (`git checkout -b feature-branch`).
3. Add your magic and commit (`git commit -m 'Add epic feature'`).
4. Push it (`git push origin feature-branch`).
5. Open a pull request and let’s make fitness history!

Keep your code clean and aligned with the project’s high-energy vibe.

---

## 🙌 Shoutouts

- [MediaPipe](https://mediapipe.dev/) for next-level pose tracking.
- [OpenCV](https://opencv.org/) for rocking the computer vision world.
- [Google Gemini AI](https://cloud.google.com/) for powering the chatbot.
- [SimpleAudio](https://simpleaudio.readthedocs.io/) for keeping the audio vibes strong.

---

## 📬 Get in Touch

Got ideas to make this project even more awesome? Drop me a line at singhshriyansh277@gmail.com or connect on [LinkedIn](https://www.linkedin.com/in/shriyansh-singh-7b0430291/)! Let’s build the future of fitness tech together! 🚀

**Time to sweat, track, and conquer!** 💪


# 🍥 Naruto All Effects — Real-Time AR Filter

Real-time Naruto visual effects triggered by hand gestures, built with Python, OpenCV and MediaPipe.

## ✨ Effects

| Gesture | Effect |
|---|---|
| ✌️ Crossed fingers | Sharingan / Rinnegan (eyes) |
| 🤟 ILY sign | Byakugan — Neji Hyuga (eyes) |
| 🖖 Vulcan sign | Gaara's Sand (body particles) |
| ✊ Fist | Chidori — Kakashi (hand lightning) |
| 👆👆 Two index fingers | Chakra Mode + Rasengan |

##  Tech Stack

- Python 3.11.2
- OpenCV — frame capture & rendering
- MediaPipe — face mesh, hand tracking, pose estimation
- NumPy — particle physics & compositing

## Run

```bash
pip install opencv-python mediapipe numpy
python most.py
```

Press `Q` or `Esc` to quit.

## Structure

Single-file project. All effects, gesture detection, and rendering logic are in `most.py`.

---

Inspired by Naruto — built for fun and computer vision practice.

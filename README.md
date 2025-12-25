# Drowsiness Detection
This project implements a real-time drowsiness detection system using computer vision and machine learning to reduce road accidents caused by driver fatigue. Instead of relying on hardware sensors or intrusive wearables, the system analyzes facial features directly from a live camera feed.

The core logic is built on eye aspect ratio (EAR) and facial landmark tracking to continuously monitor eye closure and blinking patterns. When prolonged eye closure or abnormal blinking is detected, the system classifies the driver as drowsy and immediately triggers an audio/visual alert to regain attention.

The model runs entirely on-device, making it suitable for low-cost deployment in vehicles, driver monitoring setups, or embedded systems. It works in real time with standard webcams and does not require internet connectivity.

Key Features

- Real-time face and eye detection using computer vision

- Drowsiness detection based on eye closure duration

- Instant alert system to warn the driver

- Lightweight and efficient – runs on basic hardware

Can be extended to IoT, in-car systems, or mobile apps

- Tech Stack

- Python

- OpenCV

- Dlib / MediaPipe (for facial landmarks)

- NumPy

Use Cases

- Driver safety systems

- Fleet monitoring

- Accident prevention research

- Smart vehicle applications

This project focuses on practical safety, not just theory. If you’re looking to build or extend a real-world driver monitoring system, this is a solid starting point.

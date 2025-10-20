# Real-Time Face Detection and Tracking in MATLAB

This project demonstrates real-time face detection and tracking using a webcam in MATLAB. It combines:

- **Haar Cascade Face Detection** (`vision.CascadeObjectDetector`)  
- **Feature Point Tracking** (`vision.PointTracker`)  
- **Geometric Transform Estimation** (`estimateGeometricTransform2D`)  
- **Visualization** of bounding polygons and tracked points  

The tracker automatically detects faces, tracks feature points, and handles movement or partial disappearance of the face.

## Features

- Real-time face detection and tracking  
- Automatic point tracking with geometric transformation  
- Robust: re-detects the face if tracking points are lost  
- Visualization: green bounding polygon + white `+` markers on tracked points  
- Works with any standard webcam  

## Requirements

- MATLAB R2019b or later  
- Computer Vision Toolbox  
- Webcam connected to your computer  

## Installation & Usage

1. Clone this repository:

```bash
git clone https://github.com/yourusername/RealTimeFaceTracking.git
cd RealTimeFaceTracking

2. Open MATLAB and navigate to the project folder.
3.Run the main script:
FaceTracking.m
A video window will open showing:

Green polygon around the detected face

White + markers on tracked feature points

Press ESC or close the window to stop execution.

## How It Works

1.Connects to the webcam (webcam object)

2.Detects the face using Haar Cascade

3.Finds strong feature points within the face region

4.Tracks the points using Lucas-Kanade optical flow (vision.PointTracker)

5.Updates the face bounding polygon smoothly with estimateGeometricTransform2D

6.If points are lost, the system re-detects the face automatically

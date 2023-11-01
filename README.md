# Face Recognizer

Face Recognizer is a Python-based application that can detect and recognize faces in images or via a webcam feed. It utilizes OpenCV for face detection and dlib for facial landmark extraction and recognition. This project provides a basic framework for face recognition, which can be further customized and extended for various applications.

## Features

- Face detection using OpenCV's Haar Cascade classifier.
- Facial landmark extraction using dlib's shape predictor.
- Face recognition using OpenCV's Local Binary Patterns Histograms (LBPH) recognizer.
- Real-time face recognition from a webcam feed.
- Basic example of face detection and recognition in images.

## Installation

Before running the Face Recognizer, ensure you have the required libraries installed. You can install them using pip:

```bash
pip install opencv-python dlib numpy

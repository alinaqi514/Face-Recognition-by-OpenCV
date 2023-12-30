# Face-Recognition-by-OpenCV
# Face Recognition Attendance System
# Overview
This is a simple face recognition attendance system implemented in Python using the face_recognition library and OpenCV. The system captures video from a webcam, detects faces, recognizes known faces, and maintains attendance records in a CSV file.

# Requirements
Python 3.x
OpenCV
face_recognition
numpy
Install the required libraries using the following command:
Press 'q' to exit the application.
# Configuration
Modify the known_face_encodings and known_face_names arrays to include the encoding and names of individuals you want to recognize.
Adjust the motion detection threshold (motion_threshold) as needed.
The attendance records are stored in a CSV file named with the current date.
# File Structure
attendance_system.py: The main script for face recognition and attendance tracking.
models/: Directory containing face recognition models.
# Acknowledgments
face_recognition: The face recognition library used in this project.
OpenCV: The Open Source Computer Vision Library.

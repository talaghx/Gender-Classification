# Gender Classification
A real-time gender classification system developed using ResNet18, PyTorch, OpenCV, and Python.
The system detects faces through a live camera feed and classifies them as male or female in real time.

## Features
- Real-time face detection using OpenCV
- Gender classification using ResNet18
- Displays the predicted class and confidence score
- Supports live webcam inference
- Includes the trained model file

## Technologies Used
- Python
- PyTorch
- Torchvision
- OpenCV
- ResNet18

## Project Files
- train.py – Trains the gender classification model
- camera.py – Runs real-time classification using the webcam
- best_gender_model.pth – Trained model weights

## How to Run
Install the required libraries:

bash
pip install torch torchvision opencv-python pillow

## Then run
python camera.py

Press Q to close the camera window

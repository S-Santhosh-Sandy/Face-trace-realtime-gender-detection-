*Gender Detection Using AI*
Overview
This project utilizes advanced Deep Learning and Computer Vision techniques to detect and classify gender from facial images. It leverages OpenCV for image processing and Caffe models for deep learning inference.

Features
Real-Time Detection: Processes live video streams or images to predict gender instantly.
High Accuracy: Trained on a robust dataset to ensure reliable predictions.
Easy Integration: Can be easily integrated into other applications or systems.
Customizable: Supports model tuning and parameter adjustments.
Technologies Used
Programming Language: Python

Libraries:
OpenCV
NumPy
Pygame

Deep Learning Framework: Caffe
Installation


Follow these steps to set up the project:
Clone the Repository:

git clone (https://github.com/Nishanth-2/FaceTrace-Real-Time-Gender-Detection/)
cd Gender-Detection-main
Install Required Packages:


pip install opencv-python opencv-contrib-python numpy pygame protobuf
Download Pre-trained Model:
Ensure gender_net.caffemodel and gender_deploy.prototxt are in the project directory.

Usage
Run the Detection Script:

python detect.py
Input Options:
Live Camera: Detect gender from webcam input.
Image/Video Files: Modify the script to process files instead of live streams.

Project Structure

Gender-Detection-main/
├── detect.py              # Main detection script
├── gender_net.caffemodel  # Pre-trained Caffe model
├── gender_deploy.prototxt # Model architecture file
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
Contributors
NISHANTH JS
License
This project is licensed under the MIT License. See the LICENSE file for details.

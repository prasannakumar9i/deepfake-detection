

# Deepfake Detection

This Deepfake Detection project provides a complete pipeline for identifying manipulated videos using deep learning. It includes tools for extracting faces from video frames, training a model to distinguish real from fake faces, and evaluating its accuracy. A FastAPI-powered web app allows users to upload videos and receive deepfake detection results through an intuitive interface. The system also manages user data automatically and follows a modular structure, making it easy to extend, customize, or debug.



## Features

* 🎥 **Face Extraction**: Automatically extracts faces from video frames for training/testing.
* 🤖 **Deep Learning Model**: Trains a model to differentiate between real and fake faces.
* 🧪 **Testing & Evaluation**: Evaluate model performance on a test dataset.
* 🌐 **FastAPI Web App**: Upload a video and detect deepfakes through a user-friendly web interface.
* 🗂️ **User Data Management**: Automatically manages user directories and video uploads.
* 🧩 **Modular Design**: Easy to extend and debug.

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/prasannakumar9i/deepfake-detection
cd deepfake
```

### 2. Create a Conda Environment

```bash
conda create --name deepfake python=3.11 -y
conda activate deepfake
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Create User Directory (for storing datasets and results)

```bash
python -m deepfake create-userdir
```

---

---

## 🚀 Running the Project

> ⚠️ **Ensure the dataset is placed in the `user_data/data` folder before proceeding.**

### 1. Extract Faces from Dataset

Extract frames and faces from videos for training or testing by default 80% for traing and 20% for tesing:

```bash
python -m deepfake extract 
```

### 2. Train the Model

```bash
python -m deepfake train
```

### 3. Test the Model

```bash
python -m deepfake test
```

### 4. Launch the Web App

```bash
python -m deepfake start
```

This command starts the FastAPI server. Visit `http://localhost:8000` to:

* Sign up or log in
* Upload videos
* View detection results

---
## Deepfake Detector Configuration Setup

### Create a file named config.json inside the user_data folder and add this content:


```json
{
    "models": {
        "cnn_backbone": "xception",
        "vit_backbone": "vit_tiny_patch16_224"
    },

    "model_name": "DeepfakeDetector_v2",

    "datasets": [
        "Actors",
        "Deepfakes",
        "FaceSwap",
        "DeepFakeDetection"
    ],
    "dataset_size": 500,

    "api_server": {
        "listen_ip_address": "0.0.0.0",
        "listen_port": 8080,
        "verbosity": "error",
        "jwt_secret_key": "f60d8a2b8e6307dbb2e9165db4640af1aef5b7fca99ecb63f8e61c7a9c1515f5",
        "CORS_origins": []
    },

    "verbosity": 0,
    "mode": "train"
}

# Criminal Face Detection System  

---

## Project Overview

A **real-time criminal identification system** using **deep learning-based facial recognition** with **Facenet512** and **OpenCV Haar Cascades**. The system enables:

- Adding known criminals with images and metadata  
- Detecting criminals from **uploaded images**  
- **Live webcam detection** with real-time alerts  
- High-accuracy matching using **cosine similarity > 0.6**  
- Web-based UI using **Flask**

> **Accuracy**: ~96% on same-image tests  
> **Model**: `Facenet512` (512D embeddings)  
> **Face Detection**: OpenCV Haar + Validation Filters  
> **Threshold**: 0.6 (cosine similarity)  

---

## Folder Structure

```bash
criminal-face-detection/
├── app.py                          # Flask web server
├── face_recognizer_filtered.py     # Core recognition logic (FIXED)
├── database_manager.py             # JSON-based criminal DB
├── requirements.txt                # Dependencies
├── data/
│   └── criminals/                  # Criminal images (sample)
├── database/
│   ├── criminal_db.json            # Criminal records
│   └── face_encodings.pkl          # Cached embeddings
├── static/
│   ├── results/                    # Detection outputs
│   ├── script.js                   # Frontend logic
│   └── style.css                   # UI styling
├── templates/
│   └── index.html                  # Main UI
├── uploads/                        # Temp uploaded images
└── venv/                           # Virtual environment

```


## Core Components

### 1️⃣ `face_recognizer_fixed.py`
Handles:
- Face detection with Haar cascades  
- Face embedding extraction using DeepFace (`Facenet512`)  
- Cosine similarity–based matching  
- Result visualization (red box for criminals, green for unknowns)  
- Encoding generation and caching  

### 2️⃣ `database_manager.py`
Handles:
- CRUD operations for criminal data  
- JSON-based storage and update of detection counts  

### 3️⃣ `app.py`
Flask backend for:
- Uploading test images  
- Running face detection and recognition  
- Displaying detection results on the web interface  

---

## 🛠️ Installation

### 1. Clone this repository
```bash
git clone https://github.com/yourusername/criminal-face-detection.git
cd criminal-face-detection
```

### 2.Create and activate a virtual environment
```bash

python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## ▶️ Running the Application

### 1. Start the Flask server
```bash
python app.py
```
### 2. Open the interface

Visit:
👉 http://127.0.0.1:5000

### 3. Upload an image
Upload an image containing one or more faces.
The system will:

Detect faces in the image

Compare against the criminal database

Display results:
🔴 Red Box – Criminal Detected
🟢 Green Box – Unknown Person
Output images are saved automatically to:

static/results/


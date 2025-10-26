"""
Improved Face Recognition System - FIXED VERSION
Fixes the wrong detection issues and false positives
"""

import cv2
import numpy as np
import pickle
import os
import logging
from deepface import DeepFace
import uuid
from database_manager import CriminalDatabase

class FaceRecognizerFixed:
    def __init__(self):
        self.model_name = 'Facenet512'
        self.face_detection_threshold = 0.6  # Increased threshold for better accuracy
        self.min_face_size = (50, 50)  # Minimum face size
        self.max_faces_per_image = 5
        self.encodings_file = "database/face_encodings.pkl"
        self.criminal_encodings = {}
        self.db = CriminalDatabase()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load or generate encodings
        self.load_criminal_encodings()
    
    def preprocess_image(self, image_path):
        """Preprocess image for better face detection"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"Could not load image: {image_path}")
                return None
            
            # Resize if too large (for better performance)
            height, width = image.shape[:2]
            if width > 1920 or height > 1080:
                scale = min(1920/width, 1080/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))
            
            return image
        except Exception as e:
            self.logger.error(f"Error preprocessing image: {e}")
            return None
    
    def detect_faces_accurate(self, image_path):
        """Accurate face detection using only the best method"""
        try:
            image = self.preprocess_image(image_path)
            if image is None:
                return []
            
            faces_detected = []
            
            # Use only OpenCV Haar Cascades for reliable face detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization for better detection
            gray = cv2.equalizeHist(gray)
            
            # Use frontal face cascade
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Detect faces with stricter parameters
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,  # Increased for better accuracy
                minSize=self.min_face_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            for (x, y, w, h) in faces:
                x, y, w, h = int(x), int(y), int(w), int(h)
                
                # Additional validation - check if it's actually a face
                if self.is_valid_face_region(image, x, y, w, h):
                    face_img = image[y:y+h, x:x+w]
                    face_img_resized = cv2.resize(face_img, (224, 224))
                    
                    faces_detected.append({
                        'face_image': face_img_resized,
                        'bbox': (x, y, w, h),
                        'confidence': 0.9  # High confidence for valid faces
                    })
            
            # Remove overlapping faces (keep the largest)
            faces_detected = self.remove_overlapping_faces(faces_detected)
            
            self.logger.info(f"Detected {len(faces_detected)} valid faces")
            return faces_detected
            
        except Exception as e:
            self.logger.error(f"Face detection failed: {e}")
            return []
    
    def is_valid_face_region(self, image, x, y, w, h):
        """Validate if the detected region is actually a face"""
        try:
            # Check aspect ratio (faces are roughly square)
            aspect_ratio = w / h
            if aspect_ratio < 0.7 or aspect_ratio > 1.4:
                return False
            
            # Check if region is not too small
            if w < 50 or h < 50:
                return False
            
            # Check if region is not too large (relative to image)
            img_height, img_width = image.shape[:2]
            if w > img_width * 0.8 or h > img_height * 0.8:
                return False
            
            # Check if region is within image bounds
            if x < 0 or y < 0 or x + w > img_width or y + h > img_height:
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Face validation failed: {e}")
            return False
    
    def remove_overlapping_faces(self, faces):
        """Remove overlapping face detections, keeping the largest"""
        if len(faces) <= 1:
            return faces
        
        # Sort by area (largest first)
        faces.sort(key=lambda f: f['bbox'][2] * f['bbox'][3], reverse=True)
        
        valid_faces = []
        for face in faces:
            x1, y1, w1, h1 = face['bbox']
            
            # Check if this face overlaps significantly with any valid face
            overlaps = False
            for valid_face in valid_faces:
                x2, y2, w2, h2 = valid_face['bbox']
                
                # Calculate overlap
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = overlap_x * overlap_y
                
                area1 = w1 * h1
                area2 = w2 * h2
                overlap_ratio = overlap_area / min(area1, area2)
                
                if overlap_ratio > 0.3:  # 30% overlap threshold
                    overlaps = True
                    break
            
            if not overlaps:
                valid_faces.append(face)
        
        return valid_faces
    
    def get_face_encoding_robust(self, face_image):
        """Extract face embedding with improved method"""
        try:
            if not isinstance(face_image, np.ndarray):
                self.logger.warning("Face image is not a numpy array")
                return None
            
            # Save face image temporarily
            temp_path = f"temp_face_{uuid.uuid4().hex}.jpg"
            
            # Ensure the image is in the right format
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                face_image_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
            else:
                face_image_bgr = face_image
            
            cv2.imwrite(temp_path, face_image_bgr)
            
            # Use DeepFace with strict parameters
            embedding = DeepFace.represent(
                img_path=temp_path,
                model_name=self.model_name,
                enforce_detection=False,
                detector_backend='skip'  # Skip detection since we already have face
            )
            
            if embedding and len(embedding) > 0:
                emb = np.array(embedding[0]['embedding'])
                # Normalize the embedding
                emb = emb / np.linalg.norm(emb)
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                return emb.tolist()
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        except Exception as e:
            self.logger.warning(f"Face encoding failed: {e}")
        
        return None
    
    def load_criminal_encodings(self, force_regenerate=False):
        """Load or generate criminal face encodings"""
        try:
            if os.path.exists(self.encodings_file) and not force_regenerate:
                with open(self.encodings_file, 'rb') as f:
                    self.criminal_encodings = pickle.load(f)
                self.logger.info(f"[SUCCESS] Loaded cached encodings for {len(self.criminal_encodings)} criminals")
                return
            
            # Generate new encodings
            self.generate_criminal_encodings()
            
        except Exception as e:
            self.logger.error(f"Error loading encodings: {e}")
            self.generate_criminal_encodings()
    
    def regenerate_encodings(self):
        """Force regenerate all criminal encodings"""
        self.logger.info("Force regenerating criminal encodings...")
        
        # Delete existing encodings file
        if os.path.exists(self.encodings_file):
            os.remove(self.encodings_file)
            self.logger.info("Deleted existing encodings file")
        
        # Generate new encodings
        self.generate_criminal_encodings()
    
    def generate_criminal_encodings(self):
        """Generate face encodings for all criminals"""
        self.logger.info("Generating new criminal encodings...")
        
        criminals = self.db.load_all()
        
        for criminal in criminals:
            image_path = criminal["image_path"]
            criminal_id = str(criminal["id"])
            
            if os.path.exists(image_path):
                self.logger.info(f"Processing: {criminal['name']}...")
                
                # Detect faces in criminal image
                faces = self.detect_faces_accurate(image_path)
                
                if faces:
                    # Use the first (largest) face
                    best_face = max(faces, key=lambda f: f['bbox'][2] * f['bbox'][3])
                    encoding = self.get_face_encoding_robust(best_face['face_image'])
                    
                    if encoding is not None:
                        self.criminal_encodings[criminal_id] = {
                            'encoding': encoding,
                            'name': criminal['name'],
                            'confidence': best_face['confidence']
                        }
                        self.logger.info(f"[SUCCESS] {criminal['name']} - Face encoding created")
                    else:
                        self.logger.warning(f"[FAILED] {criminal['name']} - Could not extract encoding")
                else:
                    self.logger.warning(f"[FAILED] {criminal['name']} - No face detected")
            else:
                self.logger.warning(f"[FAILED] Image not found: {image_path}")
        
        # Save encodings
        if self.criminal_encodings:
            os.makedirs(os.path.dirname(self.encodings_file), exist_ok=True)
            with open(self.encodings_file, 'wb') as f:
                pickle.dump(self.criminal_encodings, f)
            self.logger.info(f"[SUCCESS] Total criminals loaded: {len(self.criminal_encodings)}")
            self.logger.info(f"[SUCCESS] Encodings saved to: {self.encodings_file}")
    
    def find_match_accurate(self, test_encoding, threshold=None):
        """Find matching criminal with improved accuracy"""
        if threshold is None:
            threshold = self.face_detection_threshold
            
        if not self.criminal_encodings or test_encoding is None:
            return None, None, None
        
        test_encoding = np.array(test_encoding)
        matches = []
        
        for criminal_id, data in self.criminal_encodings.items():
            criminal_encoding = np.array(data['encoding'])
            
            # Calculate cosine similarity (most reliable for normalized vectors)
            cosine_sim = np.dot(test_encoding, criminal_encoding) / (
                np.linalg.norm(test_encoding) * np.linalg.norm(criminal_encoding)
            )
            
            # Calculate Euclidean distance
            euclidean_dist = np.linalg.norm(test_encoding - criminal_encoding)
            
            matches.append({
                'id': int(criminal_id),
                'name': data['name'],
                'euclidean': euclidean_dist,
                'cosine': cosine_sim
            })
        
        # Sort by cosine similarity (higher is better)
        matches.sort(key=lambda x: x['cosine'], reverse=True)
        
        if matches:
            best_match = matches[0]
            self.logger.info(f"Best match: {best_match['name']} (Cosine: {best_match['cosine']:.4f})")
            
            # Use cosine similarity for final decision
            if best_match['cosine'] > threshold:
                return best_match['id'], best_match['cosine'], best_match
        
        return None, None, None
    
    def process_image(self, image_path, output_path=None):
        """Process image and detect criminals with improved accuracy"""
        self.logger.info(f"Processing: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            self.logger.error("Error loading image!")
            return []
        
        # Detect faces with improved algorithm
        faces = self.detect_faces_accurate(image_path)
        self.logger.info(f"Detected {len(faces)} face(s)")
        
        detections = []
        
        for i, face_data in enumerate(faces):
            self.logger.info(f"Analyzing face {i+1}/{len(faces)}...")
            
            face_image = face_data['face_image']
            x, y, w, h = face_data['bbox']
            x, y, w, h = int(x), int(y), int(w), int(h)
            confidence = face_data['confidence']
            
            # Get encoding
            encoding = self.get_face_encoding_robust(face_image)
            
            if encoding is not None:
                # Find match with higher threshold for better accuracy
                criminal_id, match_score, match_details = self.find_match_accurate(encoding, threshold=0.6)
                
                if criminal_id:
                    # CRIMINAL DETECTED!
                    criminal = self.db.get_criminal_by_id(criminal_id)
                    
                    # Check if criminal exists in database
                    if criminal is None:
                        self.logger.warning(f"Criminal ID {criminal_id} not found in database")
                        continue
                    
                    confidence_percent = max(0, min(100, match_score * 100))
                    
                    self.logger.info(f"[ALERT] CRIMINAL DETECTED!")
                    self.logger.info(f"   Name: {criminal['name']}")
                    self.logger.info(f"   Crime: {criminal['crime']}")
                    self.logger.info(f"   Confidence: {confidence_percent:.1f}%")
                    
                    # Draw RED box with thicker border
                    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 4)
                    
                    # Label background
                    label = f"{criminal['name']} ({confidence_percent:.1f}%)"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    cv2.rectangle(image, (x, y-40), (x+label_size[0]+15, y), (0, 0, 255), -1)
                    cv2.putText(image, label, (x+5, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    # Alert text
                    cv2.putText(image, "CRIMINAL ALERT!", (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                    
                    # Update detection count
                    self.db.update_detection(criminal_id)
                    
                    detections.append({
                        'type': 'criminal',
                        'id': int(criminal_id),
                        'name': str(criminal['name']),
                        'crime': str(criminal['crime']),
                        'age': str(criminal['age']),
                        'gender': str(criminal['gender']),
                        'location': str(criminal['location']),
                        'confidence': f"{confidence_percent:.1f}%",
                        'match_score': f"{match_score:.4f}",
                        'bbox': [int(x), int(y), int(w), int(h)],
                        'match_details': {
                            'euclidean': float(match_details['euclidean']),
                            'cosine': float(match_details['cosine'])
                        } if match_details else None
                    })
                    
                else:
                    # Unknown person
                    self.logger.info(f"Unknown person (confidence: {confidence:.2f})")
                    
                    # Draw GREEN box
                    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.rectangle(image, (x, y-30), (x+120, y), (0, 255, 0), -1)
                    cv2.putText(image, "Unknown", (x+5, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    
                    detections.append({
                        'type': 'unknown',
                        'name': 'Unknown Person',
                        'confidence': f"{confidence:.2f}",
                        'bbox': [int(x), int(y), int(w), int(h)]
                    })
        
        # Save result image
        if output_path and output_path.strip():
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, image)
                self.logger.info(f"Result saved to: {output_path}")
            except Exception as e:
                self.logger.warning(f"Could not save result image: {e}")
        
        return detections

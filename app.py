"""
Flask Web Application - FIXED VERSION
- Proper detection results display
- Better error handling
- Criminal database management
"""

from flask import Flask, render_template, request, jsonify, Response
import os
import cv2
from werkzeug.utils import secure_filename
from database_manager import CriminalDatabase
from face_recognizer_fixed import FaceRecognizerFixed
from datetime import datetime
import threading
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Initialize components
db = CriminalDatabase()
recognizer = None
recognizer_lock = threading.Lock()

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/results', exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_recognizer():
    """Load face recognizer"""
    global recognizer
    with recognizer_lock:
        try:
            recognizer = FaceRecognizerFixed()
            print("✓ Face recognizer loaded successfully")
        except Exception as e:
            print(f"✗ Error loading recognizer: {e}")

# Load recognizer on startup
load_recognizer()

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/api/criminals', methods=['GET'])
def get_criminals():
    """Get all criminals"""
    criminals = db.load_all()
    return jsonify({'success': True, 'criminals': criminals})

@app.route('/api/criminal/add', methods=['POST'])
def add_criminal():
    """Add new criminal and reload recognizer"""
    try:
        # Get form data
        name = request.form.get('name')
        crime = request.form.get('crime')
        age = request.form.get('age')
        gender = request.form.get('gender')
        location = request.form.get('location')
        
        # Get image file
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No image selected'})
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type. Use JPG or PNG'})
        
        # Add to database (database_manager handles file saving)
        record = db.add_criminal(name, crime, age, gender, location, file)
        
        # Reload recognizer to include new criminal
        print("Reloading recognizer with new criminal...")
        with recognizer_lock:
            if recognizer is not None:
                recognizer.regenerate_encodings()
            else:
                load_recognizer()
        
        return jsonify({
            'success': True,
            'message': f'Criminal {name} added successfully! Model reloaded.',
            'criminal': record
        })
        
    except Exception as e:
        print(f"Error adding criminal: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/detect/image', methods=['POST'])
def detect_image():
    """Detect criminals in uploaded image"""
    try:
        if recognizer is None:
            return jsonify({'success': False, 'error': 'Model not loaded yet. Please wait...'})
        
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No image selected'})
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"\n{'='*60}")
        print(f"Processing uploaded image: {filename}")
        print(f"{'='*60}")
        
        # Process image with recognizer
        result_filename = f'result_{filename}'
        result_path = os.path.join('static/results', result_filename)
        
        # Get detections
        detections = recognizer.process_image(filepath, result_path)
        
        # Count criminals found
        criminals_found = sum(1 for d in detections if d.get('type') == 'criminal')
        
        print(f"\n{'='*60}")
        print(f"Detection Complete:")
        print(f"  Total faces: {len(detections)}")
        print(f"  Criminals found: {criminals_found}")
        print(f"{'='*60}\n")
        
        return jsonify({
            'success': True,
            'detections': detections,
            'result_image': f'/static/results/{result_filename}',
            'total_faces': len(detections),
            'criminals_found': criminals_found
        })
        
    except Exception as e:
        print(f"Error in detection: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/search', methods=['GET'])
def search_criminal():
    """Search criminal by name"""
    name = request.args.get('name', '')
    results = db.search_by_name(name)
    return jsonify({'success': True, 'results': results})

@app.route('/api/reload', methods=['POST'])
def reload_database():
    """Reload face recognition model"""
    try:
        load_recognizer()
        return jsonify({'success': True, 'message': 'Database reloaded successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    print("="*60)
    print("  CRIMINAL FACE DETECTION SYSTEM - WEB INTERFACE")
    print("="*60)
    print("\n🚀 Starting server...")
    print("📱 Open browser and go to: http://localhost:5000")
    print("\n✓ Server ready!\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
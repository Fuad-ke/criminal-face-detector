"""
Database Manager - FIXED VERSION
"""

import json
import os
from datetime import datetime

class CriminalDatabase:
    """Manages criminal records database"""
    
    def __init__(self, db_path="database/criminal_db.json", criminals_dir="data/criminals"):
        self.db_path = db_path
        self.criminals_dir = criminals_dir
        self.ensure_database_exists()
    
    def ensure_database_exists(self):
        """Create database file and directories if they don't exist"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs(self.criminals_dir, exist_ok=True)
        
        if not os.path.exists(self.db_path):
            with open(self.db_path, 'w') as f:
                json.dump([], f)
            print(f"[SUCCESS] Created database: {self.db_path}")
    
    def add_criminal(self, name, crime, age, gender, location, image_file):
        """Add a new criminal record with image upload"""
        criminals = self.load_all()
        
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        ext = os.path.splitext(image_file.filename)[1]
        filename = f"{len(criminals) + 1}_{name.replace(' ', '_')}_{timestamp}{ext}"
        image_path = os.path.join(self.criminals_dir, filename)
        
        # Save image
        image_file.save(image_path)
        print(f"[SUCCESS] Image saved: {image_path}")
        
        # Create record
        new_record = {
            "id": len(criminals) + 1,
            "name": name,
            "crime": crime,
            "age": age,
            "gender": gender,
            "location": location,
            "image_path": image_path,
            "date_added": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "detection_count": 0,
            "last_detected": None
        }
        
        criminals.append(new_record)
        self.save_all(criminals)
        
        print(f"[SUCCESS] Added criminal: {name} (ID: {new_record['id']})")
        return new_record
    
    def load_all(self):
        """Load all criminal records"""
        try:
            with open(self.db_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading database: {e}")
            return []
    
    def save_all(self, criminals):
        """Save all criminal records"""
        try:
            with open(self.db_path, 'w') as f:
                json.dump(criminals, f, indent=4)
            print(f"[SUCCESS] Database saved: {len(criminals)} records")
        except Exception as e:
            print(f"Error saving database: {e}")
    
    def update_detection(self, criminal_id):
        """Update detection count and timestamp"""
        criminals = self.load_all()
        for criminal in criminals:
            if criminal["id"] == criminal_id:
                criminal["detection_count"] += 1
                criminal["last_detected"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[SUCCESS] Updated detection for: {criminal['name']} (Count: {criminal['detection_count']})")
                break
        self.save_all(criminals)
    
    def get_criminal_by_id(self, criminal_id):
        """Get criminal record by ID"""
        criminals = self.load_all()
        for criminal in criminals:
            if criminal["id"] == criminal_id:
                return criminal
        return None
    
    def search_by_name(self, name):
        """Search criminal by name"""
        criminals = self.load_all()
        return [c for c in criminals if name.lower() in c["name"].lower()]
    
    def delete_criminal(self, criminal_id):
        """Delete a criminal record"""
        criminals = self.load_all()
        criminal = self.get_criminal_by_id(criminal_id)
        
        if criminal:
            # Delete image file
            if os.path.exists(criminal["image_path"]):
                os.remove(criminal["image_path"])
                print(f"[SUCCESS] Deleted image: {criminal['image_path']}")
            
            # Remove from database
            criminals = [c for c in criminals if c["id"] != criminal_id]
            self.save_all(criminals)
            print(f"[SUCCESS] Deleted criminal: {criminal['name']}")
            return True
        
        return False
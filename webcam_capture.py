import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QPixmap, QImage
import time

class WebcamCapture(QThread):
    frame_captured = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, camera_index=0, fps=30):
        super().__init__()
        self.camera_index = camera_index
        self.fps = fps
        self.is_recording = False
        self.cap = None
        self.frame_time = 1.0 / fps
        
        # Webcam settings
        self.resolution = (1920, 1080)  # Default resolution
        self.flip_horizontal = True  # Mirror webcam
        
    def initialize_camera(self):
        """Initialize camera with best settings"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                raise Exception(f"Cannot open camera {self.camera_index}")
            
            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            
            # Set FPS
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Set buffer size to reduce latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Auto exposure and focus
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Auto exposure
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Auto focus
            
            # Get actual resolution
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            print(f"📷 Webcam initialized: {actual_width}x{actual_height} @ {actual_fps} FPS")
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to initialize webcam: {str(e)}"
            print(error_msg)
            self.error_occurred.emit(error_msg)
            return False
    
    def start_recording(self):
        """Start webcam recording"""
        if self.initialize_camera():
            self.is_recording = True
            self.start()
        
    def stop_recording(self):
        """Stop webcam recording"""
        self.is_recording = False
        
    def run(self):
        """Main recording loop"""
        print("📷 Webcam capture started")
        last_time = time.time()
        
        while self.is_recording and self.cap:
            start_time = time.time()
            
            # Read frame
            ret, frame = self.cap.read()
            
            if not ret:
                print("Failed to read webcam frame")
                continue
            
            # Flip horizontally (mirror effect)
            if self.flip_horizontal:
                frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Apply filters/effects if needed
            frame_rgb = self.apply_effects(frame_rgb)
            
            # Emit frame
            self.frame_captured.emit(frame_rgb)
            
            # Control frame rate
            elapsed = time.time() - start_time
            sleep_time = max(0, self.frame_time - elapsed)
            
            if sleep_time > 0:
                time.sleep(sleep_time)
                
            # Performance monitoring
            current_time = time.time()
            if last_time > 0:
                actual_fps = 1.0 / (current_time - last_time)
                if int(current_time) % 5 == 0:  # Log every 5 seconds
                    print(f"Webcam FPS: {actual_fps:.1f}")
            last_time = current_time
        
        # Cleanup
        if self.cap:
            self.cap.release()
            self.cap = None
        
        print("📷 Webcam capture stopped")
    
    def apply_effects(self, frame):
        """Apply visual effects to webcam frame"""
        # Auto brightness/contrast adjustment
        frame = self.auto_adjust_brightness(frame)
        
        # Noise reduction
        frame = cv2.bilateralFilter(frame, 9, 75, 75)
        
        return frame
    
    def auto_adjust_brightness(self, frame):
        """Automatically adjust brightness and contrast"""
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels and convert back
        lab = cv2.merge([l, a, b])
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return frame
    
    def set_resolution(self, width, height):
        """Set webcam resolution"""
        self.resolution = (width, height)
        if self.cap:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    def set_fps(self, fps):
        """Set webcam FPS"""
        self.fps = fps
        self.frame_time = 1.0 / fps
        if self.cap:
            self.cap.set(cv2.CAP_PROP_FPS, fps)
    
    def get_available_cameras(self):
        """Get list of available cameras"""
        cameras = []
        
        # Test up to 10 camera indices
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Try to read a frame to confirm camera works
                ret, _ = cap.read()
                if ret:
                    cameras.append({
                        'index': i,
                        'name': f"Camera {i}",
                        'resolution': (
                            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        )
                    })
                cap.release()
            else:
                cap.release()
                break  # No more cameras
        
        return cameras
    
    def set_camera(self, camera_index):
        """Switch to different camera"""
        was_recording = self.is_recording
        
        if was_recording:
            self.stop_recording()
            self.wait()  # Wait for thread to finish
        
        self.camera_index = camera_index
        
        if was_recording:
            self.start_recording()


class WebcamPreview(QThread):
    """Lower FPS webcam preview for UI"""
    preview_frame = pyqtSignal(QPixmap)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, camera_index=0, fps=15):
        super().__init__()
        self.camera_index = camera_index
        self.fps = fps
        self.is_previewing = False
        self.cap = None
        self.frame_time = 1.0 / fps
        
    def start_preview(self):
        """Start webcam preview"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                raise Exception(f"Cannot open camera {self.camera_index}")
            
            # Lower resolution for preview
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            self.is_previewing = True
            self.start()
            
        except Exception as e:
            error_msg = f"Failed to start webcam preview: {str(e)}"
            print(error_msg)
            self.error_occurred.emit(error_msg)
    
    def stop_preview(self):
        """Stop webcam preview"""
        self.is_previewing = False
        
    def run(self):
        """Preview loop"""
        while self.is_previewing and self.cap:
            start_time = time.time()
            
            ret, frame = self.cap.read()
            
            if ret:
                # Flip horizontally
                frame = cv2.flip(frame, 1)
                
                # Convert to Qt format
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                
                # Emit preview
                self.preview_frame.emit(pixmap)
            
            # Control frame rate
            elapsed = time.time() - start_time
            sleep_time = max(0, self.frame_time - elapsed)
            
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # Cleanup
        if self.cap:
            self.cap.release()
            self.cap = None
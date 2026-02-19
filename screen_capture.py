import mss
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QPixmap, QImage
import time
import cv2

class ScreenCapture(QThread):
    frame_captured = pyqtSignal(np.ndarray)
    
    def __init__(self, monitor=None, fps=60):
        super().__init__()
        self.fps = fps
        self.monitor = monitor
        self.is_recording = False
        self.sct = mss.mss()
        
        # Get primary monitor if none specified
        if self.monitor is None:
            self.monitor = self.sct.monitors[1]  # Primary monitor
            
        self.frame_time = 1.0 / fps
        
    def start_recording(self):
        self.is_recording = True
        self.start()
        
    def stop_recording(self):
        self.is_recording = False
        
    def run(self):
        print(f"📹 Screen capture started at {self.fps} FPS")
        last_time = time.time()
        
        while self.is_recording:
            start_time = time.time()
            
            # Capture screen
            screenshot = self.sct.grab(self.monitor)
            
            # Convert to numpy array
            frame = np.array(screenshot)
            
            # Convert BGRA to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            
            # Emit frame
            self.frame_captured.emit(frame)
            
            # Control frame rate
            elapsed = time.time() - start_time
            sleep_time = max(0, self.frame_time - elapsed)
            
            if sleep_time > 0:
                time.sleep(sleep_time)
                
            # Performance monitoring
            current_time = time.time()
            actual_fps = 1.0 / (current_time - last_time) if last_time > 0 else 0
            last_time = current_time
            
            if int(current_time) % 5 == 0:  # Log every 5 seconds
                print(f"Screen FPS: {actual_fps:.1f}")
        
        print("📹 Screen capture stopped")
        
    def get_monitor_info(self):
        """Get information about available monitors"""
        monitors = self.sct.monitors
        monitor_info = []
        
        for i, monitor in enumerate(monitors):
            if i == 0:  # Skip the "All in One" monitor
                continue
                
            info = {
                'id': i,
                'left': monitor['left'],
                'top': monitor['top'], 
                'width': monitor['width'],
                'height': monitor['height'],
                'name': f"Monitor {i}"
            }
            monitor_info.append(info)
            
        return monitor_info
    
    def set_monitor(self, monitor_id):
        """Set which monitor to capture"""
        if monitor_id < len(self.sct.monitors):
            self.monitor = self.sct.monitors[monitor_id]
            print(f"Monitor set to: {self.monitor}")
    
    def set_capture_region(self, x, y, width, height):
        """Set custom capture region"""
        self.monitor = {
            'left': x,
            'top': y,
            'width': width,
            'height': height
        }
        print(f"Custom region set: {self.monitor}")


class ScreenPreview(QThread):
    """Lower FPS preview for UI"""
    preview_frame = pyqtSignal(QPixmap)
    
    def __init__(self, fps=15):
        super().__init__()
        self.fps = fps
        self.is_previewing = False
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]  # Primary monitor
        self.frame_time = 1.0 / fps
        
    def start_preview(self):
        self.is_previewing = True
        self.start()
        
    def stop_preview(self):
        self.is_previewing = False
        
    def run(self):
        while self.is_previewing:
            start_time = time.time()
            
            # Capture screen
            screenshot = self.sct.grab(self.monitor)
            
            # Convert to QPixmap for display
            img = QImage(
                screenshot.rgb,
                screenshot.width,
                screenshot.height,
                QImage.Format.Format_RGB888
            )
            pixmap = QPixmap.fromImage(img)
            
            # Scale down for preview (performance)
            scaled_pixmap = pixmap.scaled(400, 300, aspectRatioMode=1)  # Keep aspect ratio
            
            # Emit preview frame
            self.preview_frame.emit(scaled_pixmap)
            
            # Control frame rate
            elapsed = time.time() - start_time
            sleep_time = max(0, self.frame_time - elapsed)
            
            if sleep_time > 0:
                time.sleep(sleep_time)


class RegionSelector:
    """Helper class for selecting screen regions"""
    
    @staticmethod
    def get_window_regions():
        """Get list of open windows for selection"""
        try:
            import pygetwindow as gw
            windows = gw.getAllWindows()
            
            regions = []
            for window in windows:
                if window.visible and window.title:
                    regions.append({
                        'title': window.title,
                        'left': window.left,
                        'top': window.top,
                        'width': window.width,
                        'height': window.height
                    })
            
            return regions
        except ImportError:
            print("pygetwindow not available for window detection")
            return []
    
    @staticmethod
    def create_selection_overlay():
        """Create overlay for manual region selection"""
        # This would create a transparent overlay window
        # for drag-to-select region functionality
        pass
import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
import time
import os
from datetime import datetime

class VideoEncoder(QThread):
    progress_updated = pyqtSignal(int)  # Progress percentage
    encoding_finished = pyqtSignal(str)  # Final file path
    error_occurred = pyqtSignal(str)
    
    def __init__(self, output_path, fps=60, resolution=(1920, 1080)):
        super().__init__()
        
        self.output_path = output_path
        self.fps = fps
        self.resolution = resolution
        
        # Video writer
        self.video_writer = None
        
        # Frame queues
        self.screen_frames = []
        self.webcam_frames = []
        self.audio_data = []
        
        # Encoding settings
        self.codec = cv2.VideoWriter_fourcc(*'mp4v')  # H.264 codec
        self.quality = 95  # High quality
        
        # Processing state
        self.is_encoding = False
        self.total_frames = 0
        self.processed_frames = 0
        
    def initialize_video_writer(self):
        """Initialize video writer with optimal settings"""
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            
            # Initialize video writer
            self.video_writer = cv2.VideoWriter(
                self.output_path,
                self.codec,
                self.fps,
                self.resolution
            )
            
            if not self.video_writer.isOpened():
                raise Exception("Failed to initialize video writer")
            
            print(f"📹 Video encoder initialized: {self.resolution[0]}x{self.resolution[1]} @ {self.fps} FPS")
            return True
            
        except Exception as e:
            error_msg = f"Failed to initialize video encoder: {str(e)}"
            print(error_msg)
            self.error_occurred.emit(error_msg)
            return False
    
    def add_screen_frame(self, frame):
        """Add screen frame to encoding queue"""
        if isinstance(frame, np.ndarray):
            self.screen_frames.append(frame.copy())
        
    def add_webcam_frame(self, frame):
        """Add webcam frame to encoding queue"""
        if isinstance(frame, np.ndarray):
            self.webcam_frames.append(frame.copy())
    
    def add_audio_data(self, audio_data):
        """Add audio data to encoding queue"""
        if isinstance(audio_data, np.ndarray):
            self.audio_data.append(audio_data.copy())
    
    def start_encoding(self):
        """Start video encoding process"""
        if not self.initialize_video_writer():
            return
            
        self.is_encoding = True
        self.total_frames = len(self.screen_frames)
        self.processed_frames = 0
        
        self.start()  # Start thread
    
    def run(self):
        """Main encoding loop"""
        print("🎬 Video encoding started")
        start_time = time.time()
        
        try:
            # Process all frames
            for i, screen_frame in enumerate(self.screen_frames):
                if not self.is_encoding:
                    break
                
                # Composite frame (screen + webcam + effects)
                final_frame = self.composite_frame(screen_frame, i)
                
                # Resize to target resolution
                if final_frame.shape[:2] != (self.resolution[1], self.resolution[0]):
                    final_frame = cv2.resize(final_frame, self.resolution)
                
                # Convert RGB to BGR for OpenCV
                if len(final_frame.shape) == 3 and final_frame.shape[2] == 3:
                    final_frame = cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR)
                
                # Write frame
                self.video_writer.write(final_frame)
                
                # Update progress
                self.processed_frames += 1
                progress = int((self.processed_frames / self.total_frames) * 100)
                self.progress_updated.emit(progress)
                
                # Log progress
                if i % (self.fps * 5) == 0:  # Every 5 seconds
                    elapsed = time.time() - start_time
                    fps = i / elapsed if elapsed > 0 else 0
                    print(f"Encoding progress: {progress}% (Frame {i}/{self.total_frames}, {fps:.1f} fps)")
            
            # Finalize video
            self.finalize_video()
            
            elapsed = time.time() - start_time
            print(f"🎬 Video encoding completed in {elapsed:.1f}s")
            
            self.encoding_finished.emit(self.output_path)
            
        except Exception as e:
            error_msg = f"Encoding error: {str(e)}"
            print(error_msg)
            self.error_occurred.emit(error_msg)
        
        finally:
            self.cleanup()
    
    def composite_frame(self, screen_frame, frame_index):
        """Composite screen frame with webcam and effects"""
        # Start with screen frame
        composite = screen_frame.copy()
        
        # Add webcam overlay if available
        if frame_index < len(self.webcam_frames):
            webcam_frame = self.webcam_frames[frame_index]
            composite = self.overlay_webcam(composite, webcam_frame)
        
        # Apply post-processing effects
        composite = self.apply_effects(composite)
        
        return composite
    
    def overlay_webcam(self, screen_frame, webcam_frame):
        """Overlay webcam on screen frame"""
        if webcam_frame is None or webcam_frame.size == 0:
            return screen_frame
        
        # Webcam settings (these would come from main window settings)
        webcam_size = 200  # Default size
        position = "bottom-right"  # Default position
        shape = "Circle"  # Default shape
        
        # Resize webcam frame
        webcam_resized = cv2.resize(webcam_frame, (webcam_size, webcam_size))
        
        # Create mask based on shape
        mask = self.create_webcam_mask(webcam_size, shape)
        
        # Calculate position
        h, w = screen_frame.shape[:2]
        
        if "bottom" in position:
            y = h - webcam_size - 50
        elif "top" in position:
            y = 50
        else:
            y = (h - webcam_size) // 2
            
        if "right" in position:
            x = w - webcam_size - 50
        elif "left" in position:
            x = 50
        else:
            x = (w - webcam_size) // 2
        
        # Ensure webcam fits in screen
        x = max(0, min(x, w - webcam_size))
        y = max(0, min(y, h - webcam_size))
        
        # Apply webcam overlay with mask
        roi = screen_frame[y:y+webcam_size, x:x+webcam_size]
        
        if mask is not None:
            # Apply circular or rounded mask
            mask_inv = cv2.bitwise_not(mask)
            webcam_masked = cv2.bitwise_and(webcam_resized, webcam_resized, mask=mask)
            roi_masked = cv2.bitwise_and(roi, roi, mask=mask_inv)
            roi = cv2.add(roi_masked, webcam_masked)
        else:
            # Simple rectangular overlay
            roi = webcam_resized
        
        screen_frame[y:y+webcam_size, x:x+webcam_size] = roi
        
        return screen_frame
    
    def create_webcam_mask(self, size, shape):
        """Create mask for webcam shape"""
        mask = np.zeros((size, size), dtype=np.uint8)
        
        if shape == "Circle":
            center = size // 2
            cv2.circle(mask, (center, center), center - 5, 255, -1)
        elif shape == "Rounded":
            cv2.roundedRectangle(mask, (5, 5), (size-10, size-10), 20, 255, -1)
        else:  # Square or other shapes
            mask[:] = 255
        
        return mask
    
    def apply_effects(self, frame):
        """Apply post-processing effects"""
        # Color correction
        frame = self.auto_color_correct(frame)
        
        # Noise reduction
        frame = cv2.bilateralFilter(frame, 5, 50, 50)
        
        # Sharpening (optional)
        if hasattr(self, 'sharpen_enabled') and self.sharpen_enabled:
            frame = self.apply_sharpening(frame)
        
        return frame
    
    def auto_color_correct(self, frame):
        """Automatic color correction"""
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge and convert back
        lab = cv2.merge([l, a, b])
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return frame
    
    def apply_sharpening(self, frame):
        """Apply sharpening filter"""
        kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ])
        sharpened = cv2.filter2D(frame, -1, kernel)
        return sharpened
    
    def finalize_video(self):
        """Finalize video file"""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            
        # Add audio track if available (requires separate library like moviepy)
        if self.audio_data:
            self.merge_audio()
    
    def merge_audio(self):
        """Merge audio with video (requires moviepy)"""
        try:
            from moviepy.editor import VideoFileClip, AudioArrayClip, CompositeAudioClip
            
            # Load video
            video = VideoFileClip(self.output_path)
            
            # Convert audio data to audio clip
            audio_array = np.concatenate(self.audio_data)
            audio_clip = AudioArrayClip(audio_array, fps=44100)
            
            # Combine video and audio
            final_video = video.set_audio(audio_clip)
            
            # Export final video
            temp_path = self.output_path.replace('.mp4', '_temp.mp4')
            final_video.write_videofile(temp_path, codec='libx264', audio_codec='aac')
            
            # Replace original file
            os.replace(temp_path, self.output_path)
            
            print("🎵 Audio merged successfully")
            
        except ImportError:
            print("MoviePy not available for audio merging")
        except Exception as e:
            print(f"Error merging audio: {e}")
    
    def stop_encoding(self):
        """Stop encoding process"""
        self.is_encoding = False
    
    def cleanup(self):
        """Cleanup resources"""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        # Clear frame queues to free memory
        self.screen_frames.clear()
        self.webcam_frames.clear()
        self.audio_data.clear()


class VideoSettings:
    """Video encoding settings"""
    
    def __init__(self):
        # Resolution presets
        self.resolutions = {
            "1080p": (1920, 1080),
            "720p": (1280, 720),
            "480p": (854, 480),
            "4K": (3840, 2160)
        }
        
        # Codec options
        self.codecs = {
            "H.264": cv2.VideoWriter_fourcc(*'mp4v'),
            "H.265": cv2.VideoWriter_fourcc(*'hvc1'),
            "VP9": cv2.VideoWriter_fourcc(*'vp09')
        }
        
        # Quality presets
        self.quality_presets = {
            "Low": {"bitrate": "1M", "crf": 28},
            "Medium": {"bitrate": "5M", "crf": 23},
            "High": {"bitrate": "10M", "crf": 18},
            "Ultra": {"bitrate": "20M", "crf": 15}
        }
    
    @staticmethod
    def generate_filename(prefix="screen_recording"):
        """Generate unique filename with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}.mp4"
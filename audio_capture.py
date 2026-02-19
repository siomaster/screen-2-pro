import pyaudio
import wave
import threading
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, QTimer
import time
import queue

class AudioCapture(QThread):
    audio_data_ready = pyqtSignal(np.ndarray)
    audio_level_changed = pyqtSignal(float)  # For level indicator
    error_occurred = pyqtSignal(str)
    
    def __init__(self, sample_rate=44100, channels=2, chunk_size=1024):
        super().__init__()
        
        # Audio settings
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.format = pyaudio.paInt16
        
        # Recording state
        self.is_recording = False
        self.audio_queue = queue.Queue()
        
        # PyAudio instance
        self.audio = None
        self.stream = None
        
        # Device settings
        self.input_device_index = None
        self.output_device_index = None
        
        # Level monitoring
        self.level_smoothing = 0.1
        self.current_level = 0.0
        
        # Initialize PyAudio
        self.initialize_audio()
        
    def initialize_audio(self):
        """Initialize PyAudio and find devices"""
        try:
            self.audio = pyaudio.PyAudio()
            
            # Get default devices
            default_input = self.audio.get_default_input_device_info()
            default_output = self.audio.get_default_output_device_info()
            
            self.input_device_index = default_input['index']
            self.output_device_index = default_output['index']
            
            print(f"🎤 Default input device: {default_input['name']}")
            print(f"🔊 Default output device: {default_output['name']}")
            
        except Exception as e:
            error_msg = f"Failed to initialize audio: {str(e)}"
            print(error_msg)
            self.error_occurred.emit(error_msg)
    
    def get_audio_devices(self):
        """Get list of available audio devices"""
        if not self.audio:
            return [], []
        
        input_devices = []
        output_devices = []
        
        for i in range(self.audio.get_device_count()):
            try:
                device_info = self.audio.get_device_info_by_index(i)
                
                if device_info['maxInputChannels'] > 0:
                    input_devices.append({
                        'index': i,
                        'name': device_info['name'],
                        'channels': device_info['maxInputChannels'],
                        'sample_rate': device_info['defaultSampleRate']
                    })
                
                if device_info['maxOutputChannels'] > 0:
                    output_devices.append({
                        'index': i,
                        'name': device_info['name'],
                        'channels': device_info['maxOutputChannels'],
                        'sample_rate': device_info['defaultSampleRate']
                    })
                    
            except Exception as e:
                print(f"Error getting device {i}: {e}")
                continue
        
        return input_devices, output_devices
    
    def set_input_device(self, device_index):
        """Set microphone input device"""
        self.input_device_index = device_index
        
        if self.audio:
            try:
                device_info = self.audio.get_device_info_by_index(device_index)
                print(f"🎤 Input device set to: {device_info['name']}")
            except Exception as e:
                print(f"Error setting input device: {e}")
    
    def start_recording(self, record_microphone=True, record_system=False):
        """Start audio recording"""
        if not self.audio:
            self.error_occurred.emit("Audio not initialized")
            return
        
        try:
            # Configure stream
            if record_microphone:
                self.stream = self.audio.open(
                    format=self.format,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    input_device_index=self.input_device_index,
                    frames_per_buffer=self.chunk_size,
                    stream_callback=self.audio_callback
                )
            
            # TODO: Add system audio recording (more complex, requires loopback)
            
            self.is_recording = True
            self.start()  # Start thread
            
            print("🎤 Audio recording started")
            
        except Exception as e:
            error_msg = f"Failed to start audio recording: {str(e)}"
            print(error_msg)
            self.error_occurred.emit(error_msg)
    
    def stop_recording(self):
        """Stop audio recording"""
        self.is_recording = False
        
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            except Exception as e:
                print(f"Error stopping audio stream: {e}")
        
        print("🎤 Audio recording stopped")
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for continuous recording"""
        if status:
            print(f"Audio callback status: {status}")
        
        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        
        # Calculate audio level for UI indicator
        level = self.calculate_audio_level(audio_data)
        self.current_level = (self.current_level * (1 - self.level_smoothing) + 
                             level * self.level_smoothing)
        
        # Add to queue for processing
        self.audio_queue.put(audio_data.copy())
        
        return (in_data, pyaudio.paContinue)
    
    def run(self):
        """Main thread loop for processing audio data"""
        while self.is_recording:
            try:
                # Get audio data from queue (timeout to allow thread exit)
                audio_data = self.audio_queue.get(timeout=0.1)
                
                # Emit audio data for recording
                self.audio_data_ready.emit(audio_data)
                
                # Emit level for UI indicator
                self.audio_level_changed.emit(self.current_level)
                
            except queue.Empty:
                continue  # No audio data available
            except Exception as e:
                print(f"Error processing audio: {e}")
                break
    
    def calculate_audio_level(self, audio_data):
        """Calculate audio level (0-1) for level indicator"""
        if len(audio_data) == 0:
            return 0.0
        
        # Calculate RMS (Root Mean Square)
        rms = np.sqrt(np.mean(audio_data.astype(np.float64) ** 2))
        
        # Normalize to 0-1 range (adjust multiplier as needed)
        level = min(1.0, rms / 10000.0)
        
        return level
    
    def get_audio_level(self):
        """Get current audio level for UI"""
        return self.current_level
    
    def cleanup(self):
        """Cleanup PyAudio resources"""
        self.stop_recording()
        
        if self.audio:
            try:
                self.audio.terminate()
                self.audio = None
            except Exception as e:
                print(f"Error terminating PyAudio: {e}")


class SystemAudioCapture:
    """Capture system audio (computer output)"""
    
    def __init__(self):
        self.is_recording = False
        
        # Try different methods based on OS
        self.capture_method = self.detect_capture_method()
        
    def detect_capture_method(self):
        """Detect best method for system audio capture"""
        import platform
        system = platform.system()
        
        if system == "Windows":
            return "wasapi"  # Windows Audio Session API
        elif system == "Darwin":  # macOS
            return "soundflower"  # Requires SoundFlower or similar
        else:  # Linux
            return "pulseaudio"  # PulseAudio
    
    def start_recording(self):
        """Start system audio recording"""
        if self.capture_method == "wasapi":
            self.start_wasapi_recording()
        elif self.capture_method == "pulseaudio":
            self.start_pulseaudio_recording()
        else:
            print("System audio capture not supported on this platform")
    
    def start_wasapi_recording(self):
        """Windows system audio capture using WASAPI"""
        try:
            # This requires additional libraries like pycaw or soundcard
            # Implementation would go here
            print("WASAPI system audio capture not implemented")
            pass
        except Exception as e:
            print(f"WASAPI capture error: {e}")
    
    def start_pulseaudio_recording(self):
        """Linux system audio capture using PulseAudio"""
        try:
            # Implementation for PulseAudio capture
            print("PulseAudio system audio capture not implemented")
            pass
        except Exception as e:
            print(f"PulseAudio capture error: {e}")


class AudioLevelIndicator:
    """Audio level indicator for UI"""
    
    def __init__(self):
        self.current_level = 0.0
        self.peak_level = 0.0
        self.peak_hold_time = 0
        
    def update_level(self, level):
        """Update audio level"""
        self.current_level = level
        
        # Peak hold logic
        if level > self.peak_level:
            self.peak_level = level
            self.peak_hold_time = 30  # Hold for 30 frames
        elif self.peak_hold_time > 0:
            self.peak_hold_time -= 1
        else:
            self.peak_level *= 0.95  # Slow decay
    
    def get_level_color(self):
        """Get color based on audio level"""
        if self.current_level < 0.3:
            return "#4CAF50"  # Green
        elif self.current_level < 0.7:
            return "#FF9800"  # Orange
        else:
            return "#F44336"  # Red (clipping)
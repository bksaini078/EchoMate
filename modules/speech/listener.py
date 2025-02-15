import sounddevice as sd
import soundfile as sf
import numpy as np
import whisper
import threading
import queue
import os
import tempfile
from typing import Callable, Optional
import warnings
import json
import time
import streamlit as st

# Suppress Whisper FP16 warning
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

class AudioListener:
    def __init__(self, 
                 callback: Optional[Callable[[str], None]] = None,
                 language: str = "en",
                 sample_rate: int = 16000):
        print("Initializing AudioListener...")
        self.callback = callback
        self.language = language
        self.sample_rate = sample_rate
        self.audio_queue = queue.Queue()
        self.is_listening = False
        print("Loading Whisper model...")
        self.model = whisper.load_model("tiny")
        print("Whisper model loaded successfully")
        
    def audio_callback(self, indata, frames, time, status):
        """Callback for sounddevice to handle incoming audio data"""
        if status and status.flags.input_overflow:
            print("Audio input overflow")
        if self.is_listening:
            try:
                self.audio_queue.put(indata.copy())
                # Print audio stats occasionally
                if np.random.random() < 0.1:  # Only print ~10% of the time
                    print(f"Audio input: shape={indata.shape}, max_amplitude={np.max(np.abs(indata)):.4f}")
            except Exception as e:
                print(f"Error in audio callback: {e}")

    def process_audio(self):
        """Process audio chunks and convert to text"""
        print("Starting audio processing thread...")
        while self.is_listening:
            try:
                # Collect audio for 2 seconds before processing
                audio_data = []
                print("Collecting audio chunks...")
                for _ in range(int(self.sample_rate * 2 / 1024)):  # 2 seconds of audio
                    if not self.is_listening:
                        break
                    try:
                        chunk = self.audio_queue.get(timeout=0.5)
                        if chunk is not None and chunk.size > 0:
                            audio_data.append(chunk)
                    except queue.Empty:
                        continue

                if audio_data and self.is_listening:
                    try:
                        print(f"Processing {len(audio_data)} audio chunks...")
                        # Concatenate and process audio
                        audio_concat = np.concatenate(audio_data)
                        audio_concat = audio_concat.astype(np.float32)
                        
                        print(f"Audio data prepared: shape={audio_concat.shape}, max_amplitude={np.max(np.abs(audio_concat)):.4f}")
                        
                        # Save to temporary file
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                            sf.write(temp_audio.name, audio_concat, self.sample_rate)
                            print(f"Saved audio to temporary file: {temp_audio.name}")
                            
                            try:
                                # Transcribe using Whisper
                                print("Transcribing audio...")
                                result = self.model.transcribe(
                                    temp_audio.name,
                                    language=self.language
                                )
                                
                                if result["text"].strip():
                                    text = result["text"].strip()
                                    print(f"Transcribed text: {text}")
                                    
                                    # Update Streamlit state directly
                                    st.session_state.last_transcript = text
                                    st.session_state.last_update = time.time()
                                    
                                    # Then call callback if provided
                                    if self.callback:
                                        try:
                                            self.callback(text)
                                            print("Callback executed successfully")
                                        except Exception as e:
                                            print(f"Error in callback: {e}")
                                else:
                                    print("No text transcribed from audio")
                            except Exception as e:
                                print(f"Error transcribing audio: {e}")

                        # Clean up
                        try:
                            os.unlink(temp_audio.name)
                            print("Temporary audio file cleaned up")
                        except:
                            pass
                            
                    except Exception as e:
                        print(f"Error processing audio chunk: {e}")

            except Exception as e:
                print(f"Error in audio processing loop: {e}")
                if not self.is_listening:
                    break

    def start(self):
        """Start listening for audio input"""
        try:
            print("Starting audio listener...")
            # Get default input device
            devices = sd.query_devices()
            input_devices = [d for d in devices if d['max_input_channels'] > 0]
            
            print("\nAvailable input devices:")
            for i, device in enumerate(input_devices):
                print(f"Device {i}: {device['name']} (channels: {device['max_input_channels']})")
            
            if not input_devices:
                raise RuntimeError("No audio input devices found")
            
            default_input = sd.default.device[0]
            print(f"\nUsing input device: {devices[default_input]['name']}")
            
            self.is_listening = True
            
            # Start audio stream
            print("Starting audio stream...")
            self.stream = sd.InputStream(
                device=default_input,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=1024,
                callback=self.audio_callback
            )
            self.stream.start()
            print("Audio stream started successfully")
            
            # Start processing thread
            print("Starting processing thread...")
            self.process_thread = threading.Thread(
                target=self.process_audio,
                daemon=True,
                name="AudioProcessingThread"
            )
            self.process_thread.start()
            print("Processing thread started successfully")
            
        except Exception as e:
            self.is_listening = False
            if hasattr(self, 'stream'):
                self.stream.close()
            print(f"Error starting audio: {str(e)}")
            raise RuntimeError(f"Error starting audio: {str(e)}")

    def stop(self):
        """Stop listening for audio input"""
        print("Stopping audio listener...")
        self.is_listening = False
        if hasattr(self, 'stream'):
            try:
                self.stream.stop()
                self.stream.close()
                print("Audio stream stopped and closed")
            except Exception as e:
                print(f"Error stopping audio stream: {e}")
        if hasattr(self, 'process_thread'):
            try:
                self.process_thread.join(timeout=1.0)
                print("Processing thread stopped")
            except Exception as e:
                print(f"Error stopping processing thread: {e}")

    def set_language(self, language: str):
        """Update the language for speech recognition"""
        print(f"Setting language to: {language}")
        self.language = language

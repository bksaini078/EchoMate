from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import play
from typing import Optional
import os
import threading
import queue
import time
import asyncio

class Speaker:
    def __init__(self, voice_id: Optional[str] = "JBFqnCBsd6RMkjVDRZzb"):
        """
        Initialize the speaker with ElevenLabs configuration
        :param voice_id: Optional voice ID from ElevenLabs
        """
        load_dotenv()
        self.client = ElevenLabs()
        self.voice_id = voice_id
        self.speech_queue = queue.Queue()
        self.is_speaking = False
        self.speech_thread = None

    def process_speech_queue(self):
        """Process queued text for speech synthesis"""
        while self.is_speaking:
            try:
                try:
                    text = self.speech_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                if text:
                    try:
                        # Generate and play audio
                        audio = self.client.text_to_speech.convert(
                            text=text,
                            voice_id=self.voice_id,
                            model_id="eleven_multilingual_v2",
                            output_format="mp3_44100_128"
                        )
                        play(audio)
                        
                        # Small delay between speeches
                        time.sleep(0.5)
                        
                    except Exception as e:
                        print(f"Error generating speech: {e}")
                        
            except Exception as e:
                print(f"Error in speech processing: {e}")

    def start(self):
        """Start the speech processing"""
        if not self.is_speaking:
            self.is_speaking = True
            self.speech_thread = threading.Thread(target=self.process_speech_queue)
            self.speech_thread.start()

    def stop(self):
        """Stop the speech processing"""
        self.is_speaking = False
        if self.speech_thread:
            self.speech_thread.join(timeout=1.0)
            self.speech_thread = None

    def speak(self, text: str):
        """
        Add text to the speech queue
        :param text: Text to be converted to speech
        """
        if not text:
            return
            
        self.speech_queue.put(text)

    def set_voice(self, voice_id: str):
        """
        Update the voice ID
        :param voice_id: New voice ID from ElevenLabs
        """
        self.voice_id = voice_id

    def clear_queue(self):
        """Clear the speech queue"""
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
            except queue.Empty:
                break

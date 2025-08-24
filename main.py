import os
import threading
import pyaudio
import wave
import anthropic
from dotenv import load_dotenv
from google.cloud import texttospeech
import pygame
from io import BytesIO
import numpy as np
from faster_whisper import WhisperModel
import pyttsx3
import time
import concurrent.futures
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load environment variables
load_dotenv()

class UltraFastTranscriber:
    def __init__(self):
        # Initialize Claude client
        claude_api_key = os.getenv('CLAUDE_API_KEY')
        if not claude_api_key:
            raise ValueError("CLAUDE_API_KEY not found in environment variables")
        self.claude_client = anthropic.Anthropic(api_key=claude_api_key)
        
        # Initialize local Whisper model (GPU if available)
        print("🚀 Loading local Whisper model...")
        try:
            self.whisper_model = WhisperModel("base", device="cuda", compute_type="float16")
            print("✅ Using GPU acceleration")
        except:
            self.whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
            print("✅ Using CPU (install CUDA for GPU speedup)")
        
        # Initialize local TTS
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 200)  # Fast speaking rate
        voices = self.tts_engine.getProperty('voices')
        # Try to find a male British voice
        for voice in voices:
            if 'male' in voice.name.lower() or 'daniel' in voice.name.lower():
                self.tts_engine.setProperty('voice', voice.id)
                break
        
        # Initialize Google TTS as backup
        pygame.mixer.init()
        try:
            self.google_client = texttospeech.TextToSpeechClient()
            self.voice = texttospeech.VoiceSelectionParams(
                language_code="en-GB",
                name="en-GB-Chirp3-HD-Enceladus",
                ssml_gender=texttospeech.SsmlVoiceGender.MALE
            )
            self.audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )
            self.google_tts_available = True
        except:
            self.google_tts_available = False
            print("⚠️  Google TTS not available, using local TTS only")
        
        # Load prompt from file
        self.system_prompt = self.load_prompt_from_file()
        
        # Optimized audio settings
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000  # Optimal for Whisper
        self.chunk = 8192  # Larger chunks for speed
        self.recording = False
        self.audio_data = BytesIO()
        
        # Threading executor for parallel processing
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

    def load_prompt_from_file(self, prompt_file_path="prompt.txt"):
        """Load system prompt from a text file"""
        try:
            if os.path.exists(prompt_file_path):
                with open(prompt_file_path, 'r', encoding='utf-8') as file:
                    prompt = file.read().strip()
                    if prompt:
                        print(f"✅ Loaded character prompt from {prompt_file_path}")
                        return prompt
            print(f"⚠️  Using default prompt")
            return "You are a helpful AI assistant. Keep responses concise and conversational."
        except Exception as e:
            print(f"❌ Error reading prompt file: {e}")
            return "You are a helpful AI assistant. Keep responses concise and conversational."

    def record_audio_optimized(self):
        """Record audio directly to memory with optimized settings"""
        audio = pyaudio.PyAudio()
        
        stream = audio.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        print("🎤 Recording... Press Enter to stop.")
        self.recording = True
        audio_frames = []
        
        while self.recording:
            try:
                data = stream.read(self.chunk, exception_on_overflow=False)
                audio_frames.append(data)
            except:
                continue
        
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        return b''.join(audio_frames)
        
    def stop_recording(self):
        """Stop the current recording"""
        self.recording = False
    
    def transcribe_audio_local(self, audio_data):
        """Transcribe audio using local faster-whisper"""
        try:
            # Convert raw audio to numpy array
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Transcribe with faster-whisper
            segments, info = self.whisper_model.transcribe(
                audio_np,
                language="en",
                beam_size=1,  # Faster beam search
                best_of=1,    # Don't search multiple candidates
                temperature=0.0  # Deterministic output
            )
            
            # Extract text from segments
            text = " ".join([segment.text.strip() for segment in segments])
            return text
            
        except Exception as e:
            print(f"❌ Local transcription error: {e}")
            return None
    
    def clean_response(self, response):
        """Remove asterisk actions and text between asterisks"""
        import re
        # Remove everything between asterisks including the asterisks
        cleaned = re.sub(r'\*[^*]*\*', '', response)
        # Clean up extra whitespace
        cleaned = ' '.join(cleaned.split())
        return cleaned.strip()

    def get_claude_response(self, text):
        """Get response from Claude API using the character prompt"""
        try:
            response = self.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=150,  # Shorter for speed
                temperature=0.7,
                system=self.system_prompt,
                messages=[{"role": "user", "content": text}]
            )
            raw_response = response.content[0].text.strip()
            # Clean the response by removing asterisk actions
            cleaned_response = self.clean_response(raw_response)
            return cleaned_response
        except Exception as e:
            print(f"❌ Claude error: {e}")
            return None

    def speak_local_fast(self, text):
        """Speak using Google TTS (local TTS has issues on macOS)"""
        if not self.google_tts_available:
            return False
        
        try:
            synthesis_input = texttospeech.SynthesisInput(text=text)
            response = self.google_client.synthesize_speech(
                input=synthesis_input,
                voice=self.voice,
                audio_config=self.audio_config
            )
            
            audio_file = BytesIO(response.audio_content)
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                pygame.time.wait(50)
            return True
        except Exception as e:
            print(f"❌ TTS error: {e}")
            return False

    def speak_google_backup(self, text):
        """This is now the same as speak_local_fast"""
        return self.speak_local_fast(text)

    def parallel_process(self, transcription):
        """Process Claude response and TTS in parallel"""
        # Start Claude response
        claude_future = self.executor.submit(self.get_claude_response, transcription)
        
        # Wait for Claude response
        response = claude_future.result(timeout=10)
        
        if response:
            print(f"🤖 Character: {response}")
            
            # Try Google TTS (reliable audio output)
            print("🔊 Speaking...")
            if not self.speak_local_fast(response):
                print("❌ Speech failed")
            else:
                print("✅ Speech completed")
        
        return response

    def ultra_fast_workflow(self):
        """Ultra-optimized workflow with parallel processing"""
        start_time = time.time()
        
        # Start recording in separate thread
        record_thread = threading.Thread(target=lambda: setattr(self, 'audio_buffer', self.record_audio_optimized()))
        record_thread.start()
        
        # Wait for user input
        input()
        self.stop_recording()
        record_thread.join()
        
        if not hasattr(self, 'audio_buffer') or not self.audio_buffer:
            return None, None
        
        record_time = time.time()
        print(f"⏱️  Recording: {record_time - start_time:.2f}s")
        
        # Fast local transcription
        transcription = self.transcribe_audio_local(self.audio_buffer)
        
        if transcription:
            transcribe_time = time.time()
            print(f"📝 You said: {transcription}")
            print(f"⏱️  Transcription: {transcribe_time - record_time:.2f}s")
            
            # Parallel processing
            response = self.parallel_process(transcription)
            
            total_time = time.time()
            print(f"⏱️  Total time: {total_time - start_time:.2f}s")
            
            return transcription, response
        
        return None, None

def main():
    print("⚡ ULTRA-FAST Voice Character System")
    print("=" * 40)
    
    try:
        transcriber = UltraFastTranscriber()
        print("✅ Ultra-fast system ready!")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Install missing packages:")
        print("pip install faster-whisper pyttsx3 numpy")
        return
    
    print("\n🎯 Ready! Speak to your character...")
    
    try:
        while True:
            transcription, response = transcriber.ultra_fast_workflow()
            if transcription and response:
                print("=" * 60)
            elif transcription:
                print("❌ No response from Claude")
            else:
                print("❌ No transcription available")
                
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    finally:
        transcriber.executor.shutdown()

if __name__ == "__main__":
    main()

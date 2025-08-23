import os
import json
import tempfile
import anthropic
from dotenv import load_dotenv
import threading
import time
import pygame
from io import BytesIO
import pyaudio
import wave
from google.cloud import texttospeech
import re

# Load environment variables
load_dotenv()

# Initialize pygame mixer for audio playback
pygame.mixer.init()

class VoiceClaudeSystem:
    def stop_speaking(self):
        """Stop any ongoing speech playback."""
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass

    def __init__(self):
        # Load and validate Claude API key
        claude_api_key = os.getenv('CLAUDE_API_KEY')
        if not claude_api_key:
            raise ValueError("CLAUDE_API_KEY not found in environment variables")
        
        # Remove any whitespace/newlines that might have been copied
        claude_api_key = claude_api_key.strip()
        
        print(f"🔑 Claude API key format: {claude_api_key[:15]}...")
        print(f"🔑 Key length: {len(claude_api_key)} characters")
        
        # Validate key format
        if not claude_api_key.startswith('sk-ant-api03-'):
            print("⚠️  Warning: Claude API key should start with 'sk-ant-api03-'")
        
        self.client = anthropic.Anthropic(api_key=claude_api_key)
        
        # Initialize TTS - try local first, fallback to Google
        self.tts_engine = None
        self.use_local_tts = self._init_local_tts()
        
        if not self.use_local_tts:
            print("🌐 Using Google Cloud TTS")
            self.google_client = texttospeech.TextToSpeechClient()
            
            # Google TTS voice configuration
            self.voice = texttospeech.VoiceSelectionParams(
                language_code="en-GB",
                name="en-GB-Chirp3-HD-Enceladus",
                ssml_gender=texttospeech.SsmlVoiceGender.MALE
            )
            
            self.audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )
        
        # Audio recording settings (optimized for speed)
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000  # Standard rate, good balance of quality/speed
        self.chunk = 2048  # Larger chunk for better performance
        self.recording = False
        self.audio_frames = []
        
        # Persistent conversation memory (optimized)
        self.memory_file = "conversation_memory.json"
        self.conversation_history = self._load_conversation_memory()
        self.max_history = 10  # Keep only recent messages for faster processing

    def clean_response(self, response):
        """Remove unwanted therapeutic phrases and asterisk actions from AI responses"""
        # Remove asterisk actions like *adjusts virtual monocle*
        response = re.sub(r'\*[^*]*\*', '', response)
        
        # Remove unwanted therapeutic phrases
        bad_phrases = [
            "I'm here", "you're not alone", "remember,", "take care", 
            "feel free to reach out", "you are important", "your feelings matter"
        ]
        for phrase in bad_phrases:
            response = response.replace(phrase, "")
        
        # Clean up extra whitespace that might be left after removals
        response = ' '.join(response.split())
        
        return response.strip()

    def _load_conversation_memory(self):
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"❌ Error loading conversation memory: {e}")
        return []

    def _save_conversation_memory(self):
        try:
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"❌ Error saving conversation memory: {e}")

    def load_prompt_from_file(self, prompt_file_path="prompt.txt"):
        """Load system prompt from a text file"""
        try:
            if os.path.exists(prompt_file_path):
                with open(prompt_file_path, 'r', encoding='utf-8') as file:
                    prompt = file.read().strip()
                    if prompt:
                        print(f"✅ Loaded custom prompt from {prompt_file_path}")
                        print(f"📝 Prompt preview: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
                        return prompt
                    else:
                        print(f"⚠️  Warning: {prompt_file_path} is empty, using default prompt")
                        return None
            else:
                print(f"⚠️  Warning: {prompt_file_path} not found, using default prompt")
                return None
        except Exception as e:
            print(f"❌ Error reading prompt file: {e}")
            return None

    def record_audio(self):
        """Record audio from microphone using PyAudio until stopped"""
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
        self.audio_frames = []
        
        while self.recording:
            data = stream.read(self.chunk)
            self.audio_frames.append(data)
        
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        print("⏹️  Recording stopped.")
        
    def stop_recording(self):
        """Stop the current recording"""
        self.recording = False
    
    def save_audio_to_temp_file(self):
        """Save recorded audio to a temporary WAV file"""
        if not self.audio_frames:
            return None
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        
        try:
            audio = pyaudio.PyAudio()
            wf = wave.open(temp_file.name, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(audio.get_sample_size(self.audio_format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(self.audio_frames))
            wf.close()
            audio.terminate()
            return temp_file.name
        except Exception as e:
            print(f"❌ Audio save failed: {e}")
            return None
    
    def transcribe_audio(self, audio_file_path):
        """Transcribe audio using OpenAI Whisper API"""
        try:
            openai_api_key = os.getenv('OPENAI_API_KEY')
            if not openai_api_key:
                print("❌ OPENAI_API_KEY not found")
                return None
            
            from openai import OpenAI
            openai_client = OpenAI(api_key=openai_api_key)
            
            with open(audio_file_path, "rb") as audio_file:
                transcript = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en"
                )
            return transcript.text.strip()
            
        except Exception as e:
            print(f"❌ Transcription failed: {e}")
            return None
    
    def generate_claude_response(self, text, system_prompt=None):
        """Generate response using Claude API with persistent conversation memory."""
        try:
            # Prepare messages for Claude API format
            messages = []
            
            # Add conversation history (excluding system messages)
            for msg in self.conversation_history:
                if msg.get("role") != "system":
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            # Add current user message
            messages.append({
                "role": "user", 
                "content": text
            })
            
            # Use system prompt if provided
            system_message = system_prompt or "You are a helpful AI assistant. Respond conversationally and concisely in 1-2 sentences."
            
            # Create Claude message with speed optimizations
            response = self.client.messages.create(
                model="claude-3-5-haiku-20241022",  # Much faster model
                max_tokens=250,  # Reasonable limit to avoid cutoffs but keep responses focused
                temperature=0.1,  # Lower temperature for faster, more predictable responses
                system=system_message,
                messages=messages
            )
            
            content = response.content[0].text
            
            # Clean the response to remove unwanted phrases
            content = self.clean_response(content)
            
            # Remove common repetitive starters
            repetitive_starters = [
                "Ah, ", "Ah! ", "Ah. ", "Ah,", "Ah!", "Ah.",
                "Oh, ", "Oh! ", "Oh. ", "Oh,", "Oh!", "Oh."
            ]
            for starter in repetitive_starters:
                if content.startswith(starter):
                    content = content[len(starter):]
                    break
            
            # Capitalize first letter if needed
            if content and content[0].islower():
                content = content[0].upper() + content[1:]
            
            # Update conversation history with limits for speed
            self.conversation_history.append({"role": "user", "content": text})
            self.conversation_history.append({"role": "assistant", "content": content})
            
            # Keep conversation history manageable for faster processing
            if len(self.conversation_history) > self.max_history * 2:  # *2 because user+assistant pairs
                self.conversation_history = self.conversation_history[-(self.max_history * 2):]
            
            # Save asynchronously to not block response
            import threading
            threading.Thread(target=self._save_conversation_memory, daemon=True).start()
            
            return content
            
        except Exception as e:
            print(f"❌ Claude response error: {e}")
            return None

    def stop_speaking(self):
        """Stop any ongoing speech playback."""
        try:
            if self.use_local_tts and self.tts_engine:
                self.tts_engine.stop()
            pygame.mixer.music.stop()
        except Exception:
            pass

    def generate_speech_google(self, text):
        """Generate speech using Google Text-to-Speech API"""
        try:
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            response = self.google_client.synthesize_speech(
                input=synthesis_input,
                voice=self.voice,
                audio_config=self.audio_config
            )
            
            return response.audio_content
        except Exception as e:
            print(f"❌ Google TTS error: {e}")
            return None
    
    def play_audio_from_bytes(self, audio_bytes):
        """Play audio from bytes using pygame. Press spacebar to interrupt."""
        try:
            import sys
            import select
            import termios
            import tty
            # Load audio data into pygame
            audio_file = BytesIO(audio_bytes)
            pygame.mixer.music.load(audio_file)
            # Play the audio
            pygame.mixer.music.play()
            # Setup for non-blocking keypress
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(100)
                    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                        ch = sys.stdin.read(1)
                        if ch == ' ':
                            self.stop_speaking()
                            print("⏹️  Speech interrupted by spacebar.")
                            break
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return True
        except Exception as e:
            print(f"❌ Audio playback error: {e}")
            return False
    
    def _init_local_tts(self):
        """Initialize local TTS engine if available"""
        try:
            import pyttsx3
            self.tts_engine = pyttsx3.init()
            
            # Configure voice settings
            voices = self.tts_engine.getProperty('voices')
            
            # Try to find a male British voice, fallback to any male voice
            preferred_voice = None
            for voice in voices:
                voice_name = voice.name.lower()
                voice_id = voice.id.lower()
                
                # Look for British male voices
                if ('british' in voice_name or 'uk' in voice_name or 'daniel' in voice_name) and 'male' in voice_name:
                    preferred_voice = voice.id
                    break
                # Fallback to any male voice
                elif 'male' in voice_name or 'david' in voice_name or 'alex' in voice_name:
                    preferred_voice = voice.id
            
            if preferred_voice:
                self.tts_engine.setProperty('voice', preferred_voice)
                print(f"🗣️  Using local TTS voice: {preferred_voice}")
            
            # Set speech rate and volume for speed
            self.tts_engine.setProperty('rate', 200)  # Faster speech rate
            self.tts_engine.setProperty('volume', 0.95)
            
            print("⚡ Local TTS initialized successfully (SPEED MODE!)")
            return True
            
        except ImportError:
            print("📦 pyttsx3 not installed, using Google TTS")
            print("💡 For faster TTS: pip install pyttsx3")
            return False
        except Exception as e:
            print(f"⚠️  Local TTS init failed: {e}, using Google TTS")
            return False

    def speak_text(self, text):
        """Generate and play speech using local TTS"""
        if self.use_local_tts:
            try:
                import threading
                def speak_async():
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                
                speak_thread = threading.Thread(target=speak_async, daemon=True)
                speak_thread.start()
                return
            except Exception as e:
                print(f"❌ TTS failed: {e}")
        
        # Fallback to Google TTS
        self._speak_with_google_tts(text)
    
    def _speak_with_google_tts(self, text):
        """Generate and play speech using Google TTS (original method)"""
        # Generate speech
        audio_bytes = self.generate_speech_google(text)
        
        if audio_bytes:
            # Play the audio
            success = self.play_audio_from_bytes(audio_bytes)
            if success:
                print("✅ Google TTS speech playback completed.")
            else:
                print("❌ Failed to play speech audio.")
        else:
            print("❌ Failed to generate speech.")
    
    def process_voice_input(self, system_prompt=None):
        """Complete workflow: record -> transcribe -> generate response -> speak"""
        # Start recording in a separate thread
        record_thread = threading.Thread(target=self.record_audio)
        record_thread.start()
        # Wait for user to press Enter to stop recording
        input()
        self.stop_recording()
        record_thread.join()
        
        # Save audio to temporary file
        audio_file = self.save_audio_to_temp_file()
        if not audio_file:
            print("❌ No audio recorded")
            return None, None
        
        try:
            # Transcribe
            transcription = self.transcribe_audio(audio_file)
            if not transcription:
                print("❌ Transcription failed")
                return None, None
                
            print(f"You: {transcription}")
            
            # Generate response
            response = self.generate_claude_response(transcription, system_prompt)
            if not response:
                print("❌ Response generation failed")
                return None, None
                
            print(f"Claude: {response}")
            
            # Speak response
            self.speak_text(response)
            return transcription, response
            
        finally:
            # Clean up temporary file
            if os.path.exists(audio_file):
                os.unlink(audio_file)
        
        return None, None

def main():
    print("🎤 Voice-to-Claude System with Google TTS (en-GB-Chirp3-HD-Enceladus)")
    print("=" * 50)
    
    # Initialize the system
    try:
        voice_claude = VoiceClaudeSystem()
        print("✅ System initialized successfully!")
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        print("Make sure you have:")
        print("- CLAUDE_API_KEY in your .env file")
        print("- OPENAI_API_KEY in your .env file (for Whisper transcription)")
        print("- Google Cloud credentials set up")
        print("- Required packages: pip3 install anthropic pyaudio python-dotenv pygame google-cloud-texttospeech")
        return
    
    # Load system prompt from file or use default
    custom_prompt = voice_claude.load_prompt_from_file("prompt.txt")
    
    if custom_prompt:
        system_prompt = custom_prompt
    else:
        system_prompt = "You are a helpful AI assistant. Respond conversationally and concisely in 1-2 sentences."
        print("📝 Using default system prompt")
    
    print("\n🎯 Ready! Simply speak when recording starts, then press Enter to stop.")
    print("Press Ctrl+C to quit anytime.\n")
    
    conversation_count = 0
    
    try:
        while True:
            conversation_count += 1
            print(f"\n--- Conversation {conversation_count} ---")
            
            transcription, response = voice_claude.process_voice_input(system_prompt)
            
            print("🔄 Ready for next exchange...\n")
            time.sleep(0.5)  # Brief pause before next round
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Thanks for chatting!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()

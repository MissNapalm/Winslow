import os
import sys
import threading
import pyaudio
import anthropic
from dotenv import load_dotenv
from io import BytesIO
import numpy as np
from faster_whisper import WhisperModel
import time
import concurrent.futures
import warnings
import subprocess
import platform
import shutil
import json
import re

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load environment variables
load_dotenv()

# ---------- Typewriter printer (yellow) ----------
def typewriter_print(text, delay=0.01, color="\033[93m"):
    """Print text with a typewriter effect in yellow by default (ANSI)."""
    try:
        sys.stdout.write(color)
        sys.stdout.flush()
    except Exception:
        pass
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)
    # reset color + newline
    try:
        sys.stdout.write("\033[0m\n")
    except Exception:
        sys.stdout.write("\n")
    sys.stdout.flush()

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
        except Exception:
            self.whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
            print("✅ Using CPU (install CUDA for GPU speedup)")
        
        # Initialize TTS
        self.setup_tts()
        
        # Load base system prompt from file
        self.base_system_prompt = self.load_prompt_from_file()

        # Conversation memory (persisted)
        self.history = []             # list of {"role": "user"|"assistant", "content": str}
        self.running_summary = ""     # rolling summary of older context
        self.max_history_chars = 14000
        self.memory_path = "conversation.json"
        self._load_memory()
        
        # Optimized audio settings
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 8192
        self.recording = False
        self.audio_data = BytesIO()
        
        # Threading executor
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

        # Keyboard raw mode support (for spacebar)
        self._raw_supported = self._detect_raw_mode_support()

    # ---------- Memory management ----------
    def _load_memory(self):
        try:
            if os.path.exists(self.memory_path):
                with open(self.memory_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.history = data.get("history", [])
                self.running_summary = data.get("running_summary", "")
                print("🧠 Loaded conversation memory.")
        except Exception as e:
            print(f"⚠️ Could not load memory: {e}")

    def _save_memory(self):
        try:
            with open(self.memory_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"history": self.history, "running_summary": self.running_summary},
                    f, ensure_ascii=False, indent=2
                )
        except Exception as e:
            print(f"⚠️ Could not save memory: {e}")

    def _maybe_summarize_history(self):
        """When history gets big, summarize the oldest half into running_summary."""
        try:
            serialized = json.dumps(self.history, ensure_ascii=False)
            if len(serialized) <= self.max_history_chars or len(self.history) < 6:
                return

            cut = max(3, len(self.history) // 2)
            old_chunk = self.history[:cut]
            self.history = self.history[cut:]

            prompt = (
                "Summarize the prior conversation turns below into 8–12 concise bullet points. "
                "Capture user preferences, facts about the user, ongoing tasks, decisions, and unresolved threads. "
                "Keep it neutral and compact.\n\n" + json.dumps(old_chunk, ensure_ascii=False)
            )
            resp = self.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=300,
                temperature=0.2,
                system="You are a precise summarizer.",
                messages=[{"role": "user", "content": prompt}]
            )
            summary = resp.content[0].text.strip()
            self.running_summary += (("\n" if self.running_summary else "") + summary)
            print("🧠 Summarized older history.")
        except Exception as e:
            print(f"⚠️ Could not summarize: {e}")

    def _reset_memory(self):
        self.history = []
        self.running_summary = ""
        self._save_memory()
        print("🧼 Memory reset.")

    # ---------- TTS ----------
    def setup_tts(self):
        self.tts_available = False
        if platform.system() == 'Darwin':
            if shutil.which('say'):
                self.tts_available = True
                print("✅ macOS system TTS available")
        elif platform.system() == 'Linux':
            if shutil.which('espeak'):
                self.tts_available = True
                print("✅ Linux espeak TTS available")

    def clean_text_for_tts(self, text):
        """Remove visual elements that don't speak well but keep the text readable."""
        if not text:
            return ""
        
        # Remove common emoticons and visual elements that TTS mangles
        tts_text = text
        
        # Remove bracketed emoticons like >:], :], [smirk], etc.
        tts_text = re.sub(r'>:\]', '', tts_text)
        tts_text = re.sub(r':\]', '', tts_text) 
        tts_text = re.sub(r'\[.*?\]', '', tts_text)
        tts_text = re.sub(r'<.*?>', '', tts_text)
        
        # Remove other common visual markers
        tts_text = re.sub(r'[>]{2,}', '', tts_text)  # Remove >> markers
        tts_text = re.sub(r'[*]{1,2}[^*]*[*]{1,2}', '', tts_text)  # Remove *action* markers
        
        # Clean up extra whitespace
        tts_text = ' '.join(tts_text.split()).strip()
        
        return tts_text

    def speak_system(self, text):
        if not self.tts_available:
            # Still return True to avoid blocking workflows that wait for TTS
            print(f"🔊 Character would say: {text}")
            return True
        
        # Clean text for TTS while keeping original for display
        tts_text = self.clean_text_for_tts(text)
        
        try:
            if platform.system() == 'Darwin':
                subprocess.run(['say', '-v', 'Jamie (Premium)', '-r', '180', tts_text], check=True)
            elif platform.system() == 'Linux':
                subprocess.run(['espeak', '-s', '160', '-p', '40', tts_text], check=True)
            return True
        except Exception as e:
            print(f"❌ System TTS error: {e}")
            print(f"🔊 Character would say: {tts_text}")
            return False

    def speak_async(self, text):
        """Start OS TTS in the background so it overlaps with the typewriter."""
        t = threading.Thread(target=self.speak_system, args=(text,), daemon=True)
        t.start()
        return t

    # ---------- Prompt ----------
    def load_prompt_from_file(self, prompt_file_path="prompt.txt"):
        try:
            if os.path.exists(prompt_file_path):
                with open(prompt_file_path, 'r', encoding='utf-8') as file:
                    prompt = file.read().strip()
                    if prompt:
                        print(f"✅ Loaded character prompt from {prompt_file_path}")
                        return prompt
            return "You are a helpful AI assistant. Keep responses concise and conversational."
        except Exception as e:
            print(f"❌ Error reading prompt file: {e}")
            return "You are a helpful AI assistant. Keep responses concise and conversational."

    # ---------- Audio + Spacebar interrupt ----------
    def _detect_raw_mode_support(self):
        if platform.system() == "Windows":
            return True  # we'll use msvcrt
        # POSIX: require a TTY
        return sys.stdin.isatty()

    # POSIX raw mode helpers
    def _posix_raw_reader(self, stop_flag):
        """Read single chars in raw mode on a background thread; stop on SPACE/ENTER."""
        import termios, tty, select
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)  # cbreak so we get chars immediately
            while self.recording and not stop_flag.is_set():
                rlist, _, _ = select.select([sys.stdin], [], [], 0.02)
                if rlist:
                    ch = sys.stdin.read(1)
                    if ch in (' ', '\n', '\r'):
                        self.stop_recording()
                        stop_flag.set()
                        break
        except Exception:
            pass
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            except Exception:
                pass

    def record_audio_optimized(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )

        # UX hint
        if self._raw_mode_is_windows():
            print("🎤 Recording... Press SPACE (or Enter) to stop.")
        elif self._raw_supported:
            print("🎤 Recording... Press SPACE (or Enter) to stop.")
        else:
            print("🎤 Recording... Press Enter to stop (raw keys unavailable).")

        self.recording = True
        audio_frames = []

        # Start a raw key listener where possible
        stop_flag = threading.Event()
        key_thread = None

        if self._raw_mode_is_windows():
            import msvcrt
        elif self._raw_supported:
            key_thread = threading.Thread(target=self._posix_raw_reader, args=(stop_flag,), daemon=True)
            key_thread.start()
        else:
            def _enter_waiter():
                try:
                    input()
                except Exception:
                    pass
                self.stop_recording()
                stop_flag.set()
            key_thread = threading.Thread(target=_enter_waiter, daemon=True)
            key_thread.start()

        try:
            while self.recording:
                if self._raw_mode_is_windows():
                    import msvcrt
                    if msvcrt.kbhit():
                        try:
                            ch = msvcrt.getch()
                            if ch in (b' ', b'\r', b'\n'):
                                self.stop_recording()
                                break
                        except Exception:
                            pass

                try:
                    data = stream.read(self.chunk, exception_on_overflow=False)
                    audio_frames.append(data)
                except Exception:
                    continue
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()
            # ensure key thread cleaned up
            stop_flag.set()
            if key_thread and key_thread.is_alive():
                try:
                    key_thread.join(timeout=0.1)
                except Exception:
                    pass

        return b''.join(audio_frames)

    def _raw_mode_is_windows(self):
        return platform.system() == "Windows"

    def stop_recording(self):
        self.recording = False
    
    def transcribe_audio_local(self, audio_data):
        try:
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Add initial prompt to discourage transcribing punctuation as words
            initial_prompt = "Hello, how are you today? I'm doing well, thanks for asking."
            
            segments, info = self.whisper_model.transcribe(
                audio_np,
                language="en",
                beam_size=1,
                best_of=1,
                temperature=0.0,
                initial_prompt=initial_prompt
            )
            text = " ".join([segment.text.strip() for segment in segments])
            
            # Clean up common transcription artifacts
            text = self.clean_transcription(text)
            
            return text
        except Exception as e:
            print(f"❌ Local transcription error: {e}")
            return None

    def clean_transcription(self, text: str) -> str:
        """Clean transcription artifacts like spoken punctuation."""
        if not text:
            return ""
        
        cleaned = text.strip()
        
        # Remove common transcription artifacts where punctuation is transcribed as words
        punctuation_words = [
            (r'\bcomma\b', ','),
            (r'\bperiod\b', '.'),  
            (r'\bquestion mark\b', '?'),
            (r'\bexclamation mark\b', '!'),
            (r'\bexclamation point\b', '!'),
            (r'\bcolon\b', ':'),
            (r'\bsemicolon\b', ';'),
            (r'\bdash\b', '-'),
            (r'\bhyphen\b', '-'),
            (r'\bquote\b', '"'),
            (r'\bunquote\b', '"'),
            (r'\bopen paren\b', '('),
            (r'\bclose paren\b', ')'),
            (r'\bopen parenthesis\b', '('),
            (r'\bclose parenthesis\b', ')'),
        ]
        
        # Apply replacements
        for word_pattern, punctuation in punctuation_words:
            cleaned = re.sub(word_pattern, punctuation, cleaned, flags=re.IGNORECASE)
        
        # Remove standalone punctuation words at the beginning
        cleaned = re.sub(r'^(comma|period|question mark|exclamation|colon|semicolon)\s+', '', cleaned, flags=re.IGNORECASE)
        
        # Clean up extra spaces around punctuation
        cleaned = re.sub(r'\s+([,.!?;:])', r'\1', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned

    # ---------- Helpers ----------
    def clean_response(self, response: str) -> str:
        """Enhanced cleaning to handle common speech artifacts and remove all emoticons."""
        if not response:
            return ""
        
        cleaned = response.strip()
        
        # Remove ALL emoticons and visual elements
        # Remove bracketed emoticons like >:], :], [smirk], etc.
        cleaned = re.sub(r'>:\]', '', cleaned)
        cleaned = re.sub(r':\]', '', cleaned)
        cleaned = re.sub(r':\)', '', cleaned)
        cleaned = re.sub(r':\(', '', cleaned)
        cleaned = re.sub(r';-?\)', '', cleaned)  # ;) or ;-)
        cleaned = re.sub(r':-?\)', '', cleaned)  # :) or :-)
        cleaned = re.sub(r':-?\(', '', cleaned)  # :( or :-(
        cleaned = re.sub(r':-?[DdPpOo]', '', cleaned)  # :D, :P, :O, etc.
        cleaned = re.sub(r'\[.*?\]', '', cleaned)  # [smirk], [smile], etc.
        cleaned = re.sub(r'<.*?>', '', cleaned)   # <grin>, <wink>, etc.
        
        # Remove action asterisks like *smiles*
        cleaned = re.sub(r'\*[^*]*\*', '', cleaned)
        
        # Remove other visual markers
        cleaned = re.sub(r'[>]{2,}', '', cleaned)  # Remove >> markers
        cleaned = re.sub(r'[~]{2,}', '', cleaned)  # Remove ~~ markers
        
        # Remove common speech artifacts and unwanted prefixes
        prefixes_to_remove = [
            r'^,\s*',           # Remove leading comma
            r'^comma\s*',       # Remove "comma" at start
            r'^um\s*',          # Remove "um"
            r'^uh\s*',          # Remove "uh"
            r'^well\s*',        # Remove "well"
            r'^so\s*',          # Remove "so" at start
            r'^okay\s*',        # Remove "okay" at start (when inappropriate)
        ]
        
        for prefix in prefixes_to_remove:
            cleaned = re.sub(prefix, '', cleaned, flags=re.IGNORECASE)
        
        # Clean up extra whitespace and punctuation spacing
        cleaned = re.sub(r'\s+([,.!?;:])', r'\1', cleaned)  # Fix spacing before punctuation
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()      # Normalize all whitespace
        
        # Ensure first letter is capitalized if it's a sentence
        if cleaned and cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]
        
        return cleaned

    def _build_system_prompt(self) -> str:
        """Combine base system prompt + running memory into a single top-level system string."""
        base_prompt = self.base_system_prompt
        
        # Add instruction to avoid unwanted prefixes
        speech_instruction = (
            "\n\nIMPORTANT: Respond naturally and conversationally. "
            "Do NOT start responses with words like 'comma', 'um', 'uh', 'well', or other speech artifacts. "
            "Jump straight into your response content."
        )
        
        if self.running_summary:
            return (
                f"{base_prompt}{speech_instruction}\n\n"
                f"--- Memory (summarized context) ---\n"
                f"{self.running_summary}\n"
                f"--- End Memory ---"
            )
        return f"{base_prompt}{speech_instruction}"

    def _maybe_handle_voice_command(self, text: str) -> bool:
        """Return True if a local command was handled (e.g., reset memory)."""
        if not text:
            return False
        t = text.strip().lower()
        if any(cmd in t for cmd in ["reset memory", "clear memory", "forget everything", "wipe memory"]):
            self._reset_memory()
            print("🗑️  Memory cleared by voice command.")
            return True
        return False

    # ---------- Claude ----------
    def get_claude_response(self, text):
        # Optional local command (e.g., "reset memory")
        if self._maybe_handle_voice_command(text):
            return "Okay — I've cleared our memory."

        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"🤖 Getting Claude response (attempt {attempt + 1}/{max_retries})...")

                # Append new user turn and maybe summarize
                self.history.append({"role": "user", "content": text})
                self._maybe_summarize_history()

                # Build messages: ONLY user/assistant roles
                conv = [
                    m for m in self.history[-20:]
                    if isinstance(m, dict) and m.get("role") in ("user", "assistant")
                ]

                # Top-level system carries base prompt + running memory + speech instructions
                system_str = self._build_system_prompt()

                response = self.claude_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=200,
                    temperature=0.7,
                    system=system_str,   # ✅ top-level system with speech instructions
                    messages=conv        # ✅ only 'user'|'assistant'
                )
                raw = response.content[0].text.strip()
                cleaned = self.clean_response(raw)

                # Double-check for comma prefix after cleaning
                if cleaned.lower().startswith('comma'):
                    cleaned = cleaned[5:].strip()  # Remove "comma" and any following space
                    if cleaned and cleaned[0].islower():
                        cleaned = cleaned[0].upper() + cleaned[1:]

                # Save assistant reply and persist
                self.history.append({"role": "assistant", "content": cleaned})
                self._save_memory()

                return cleaned
            except Exception as e:
                print(f"❌ Claude error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
        return "Sorry, I'm having trouble connecting right now. Can you try again?"

    def parallel_process(self, transcription):
        """Get reply, start TTS + typewriter together, and WAIT until TTS finishes before returning."""
        try:
            reply = self.executor.submit(self.get_claude_response, transcription).result(timeout=45)
            if reply:
                # Start TTS immediately (non-blocking) so it overlaps with typewriter
                tts_thread = self.speak_async(reply)
                # Yellow typewriter prints in parallel with TTS
                typewriter_print(f"🤖 Character: {reply}")
                # 🔒 IMPORTANT: wait for TTS to finish before starting next cycle
                if tts_thread is not None:
                    tts_thread.join()
            return reply
        except concurrent.futures.TimeoutError:
            fallback = "Sorry, I'm taking too long to think. Can you try again?"
            tts_thread = self.speak_async(fallback)
            typewriter_print(f"🤖 Character: {fallback}")
            if tts_thread is not None:
                tts_thread.join()
            return fallback
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return None

    # ---------- Workflow ----------
    def ultra_fast_workflow(self):
        # Start recording directly; stop via SPACE or Enter (raw), or Enter fallback
        self.audio_buffer = self.record_audio_optimized()

        if not self.audio_buffer:
            return None, None

        raw_transcription = self.transcribe_audio_local(self.audio_buffer)
        if raw_transcription:
            print(f"📝 Raw transcription: {raw_transcription}")
            print(f"📝 You said: {raw_transcription}")
            # parallel_process waits for TTS to complete before returning
            response = self.parallel_process(raw_transcription)
            print(f"🔍 Claude's raw response: {repr(response)}")
            return raw_transcription, response
        return None, None

def main():
    print("⚡ ULTRA-FAST Voice Character System")
    print("=" * 40)
    try:
        transcriber = UltraFastTranscriber()
        print("✅ Ultra-fast system ready!")
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    if transcriber.tts_available:
        print("🔊 System TTS enabled")
    else:
        print("⚠️  No TTS available - text output only")
        if platform.system() == 'Linux':
            print("   • sudo apt-get install espeak")
        elif platform.system() not in ['Darwin', 'Linux']:
            print("   • Use macOS or Linux for system TTS")

    print("\n🎯 Ready! Speak — press SPACE (or Enter) to stop.")
    try:
        while True:
            try:
                transcription, response = transcriber.ultra_fast_workflow()
                if transcription and response:
                    print("=" * 60)
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error in workflow: {e}")
                continue
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    finally:
        try:
            transcriber.executor.shutdown(wait=False)
        except Exception:
            pass

if __name__ == "__main__":
    main()

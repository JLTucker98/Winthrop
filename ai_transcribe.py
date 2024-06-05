#Josh Tucker, Copyright 2024
#This software is provided under the MIT License

import keyboard
import time
import pyaudio
import threading
import whisper
import io
import wave
import numpy as np
import requests
import pyperclip
import winsound

is_recording = False
run_prog = True

# Initialize infrastructure
p = pyaudio.PyAudio()
print("loading whisper")
model = whisper.load_model("medium.en")
system_prompt = 'You are a patent attorney.  Keep your response short and to the point. Just give the response.  For instance, if responding to an email, just respond with the body of the email.  Do not use headings or bullets.  '  #consider making this a system message

def get_ai_response(prompt):
    api_url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3",  # Specify the model name
        "prompt": prompt,  # Your prompt
        "stream": False,
        "use_mlock": True,
    }  #consider adding keep_alive: 0 at shutdown to free vram
    response = requests.post(api_url, json=payload)
    if response.status_code == 200:
        generated_text = response.json().get("response", "")
        return generated_text
    else:
        return (f"Request failed with status code {response.status_code}")

def add_text_from_clipboard_if_referenced(text):
    words = text.lower().strip(" ,.?!").split()
    if len(words)>3:
        print(words[-3:])
        if words[-3:] == ['in', 'my', 'clipboard']:
            print('appending clipboard to prompt')
            return text.replace('in my clipboard', ': ' + pyperclip.paste())
        else: return text
    else: return text


#function to route to ai
def process_if_ai_invoked(transcript):
    first_word = transcript.lstrip().partition(' ')[0].lower().rstrip(" ,.")
    if first_word == "winthrop": #AI named winthrop invoked
        prompt_text = system_prompt + transcript.replace("winthrop", "", 1).lstrip(" ,.")
        prompt_plus_clpbrd = add_text_from_clipboard_if_referenced(prompt_text)
        print('ai invoked: prompt text = ' + prompt_plus_clpbrd)
        return get_ai_response(prompt_plus_clpbrd)
    else:
        print('ai not invoked')
        return transcript

# Function to create an in-memory WAV object from pyaudio frames
def create_memory_wav(frames, rate=16000):
    audio_buffer = io.BytesIO() # Create an in-memory bytes buffer
    with wave.open(audio_buffer, 'wb') as wf:
        wf.setnchannels(1)  # Mono channel
        wf.setsampwidth(2)  # 16-bit samples
        wf.setframerate(rate)  # Sample rate
        wf.writeframes(b''.join(frames))  # Write the audio frames
    audio_buffer.seek(0) # Reset the buffer to the beginning (seek to start)
    return audio_buffer

# Function to convert in-memory WAV to format that whisper wants
def wav_to_numpy(wav_file):
    with wave.open(wav_file, 'rb') as wf:
        raw_data = wf.readframes(wf.getnframes()) #Read raw audio data
        audio_np = np.frombuffer(raw_data, dtype=np.int16)#Convert raw data to numpy array 
        audio_float_np = audio_np.astype(np.float32) / 32768  # Normalize to [-1, 1]
        return audio_float_np

def convert_speech_to_text(frames):
    memory_wav = create_memory_wav(frames) # Create an in-memory WAV object from frames
    audio_np = wav_to_numpy(memory_wav) # Convert WAV data to numpy array
    transcription_result = model.transcribe(audio_np)
    AI_result_if_invoked = process_if_ai_invoked(transcription_result['text'])
    print('\nresult:')
    print(AI_result_if_invoked)
    pyperclip.copy(AI_result_if_invoked)
    winsound.Beep(250, 50)
        
def start_recording():
    global is_recording
    if not is_recording:
        winsound.Beep(350, 150)
        is_recording = True
        print("Recording started.")
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=16000,
                        input=True,
                        frames_per_buffer=1024)
        frames = []
        while is_recording: # The recording loop
            data = stream.read(1024)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        print("Recording stopped.")
        convert_speech_to_text(frames)
    else:
        winsound.Beep(550, 800)

def stop_recording():
    global is_recording
    is_recording = False

def stop_program():
    global run_prog
    run_prog = False

# Hotkeys for starting and stopping recording
hk_start_recording = "Ctrl+Alt+Z"
hk_stop_recording = "Ctrl+Alt+X"
hk_stop_program = "Ctrl+Alt+V"

# Function to start recording in a separate thread
def start_recording_thread():
    recording_thread = threading.Thread(target=start_recording)
    recording_thread.start()

# Register the hotkeys
keyboard.add_hotkey(hk_start_recording, start_recording_thread)
keyboard.add_hotkey(hk_stop_recording, stop_recording)
keyboard.add_hotkey(hk_stop_program, stop_program)
print("Hotkeys registered:")
print("  Start recording: ", hk_start_recording)
print("  Stop recording:  ", hk_stop_recording)
print("  Stop program:  ", hk_stop_program)

# Loop to keep the program running
try:
    while run_prog:
        time.sleep(0.1)  # Allow hotkey handling
except KeyboardInterrupt:
    pass

# Clean up on exit
keyboard.remove_hotkey(hk_start_recording)
keyboard.remove_hotkey(hk_stop_recording)
keyboard.remove_hotkey(hk_stop_program)
p.terminate()


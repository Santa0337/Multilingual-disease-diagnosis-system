# import streamlit as st
# import pyaudio
# import json
# from vosk import Model, KaldiRecognizer
# import threading

# # Function to load the Vosk model based on the selected language
# def load_model(language):
#     if language == 'Telugu':
#         return Model("model/vosk-model-small-te-0.42")
#     elif language == 'Hindi':
#         return Model("model/vosk-model-small-hi-0.22")
#     else:
#         raise ValueError("Unsupported language")

# # Function to start listening and update the text variable
# def listen_for_speech(language):
#     model = load_model(language)
#     rec = KaldiRecognizer(model, 16000)

#     # Initialize PyAudio
#     p = pyaudio.PyAudio()
#     stream = p.open(format=pyaudio.paInt16,
#                     channels=1,
#                     rate=16000,
#                     input=True,
#                     frames_per_buffer=8000)
#     stream.start_stream()

#     text = ""
#     while True:
#         data = stream.read(4000, exception_on_overflow=False)
#         if rec.AcceptWaveform(data):
#             result = json.loads(rec.Result())
#             text = result.get("text", "")
#             if text:
#                 break  # Stop listening after capturing the first sentence

#     stream.stop_stream()
#     stream.close()
#     p.terminate()
#     return text

# # Streamlit UI
# st.title("Multilingual Speech Recognition")
# st.write("Click the button below to start listening and capture a sentence.")

# # Language selection dropdown
# language = st.selectbox("Select language", ("Telugu", "Hindi"))

# if st.button("Start Listening"):
#     with st.spinner(f"Listening in {language}..."):
#         text = listen_for_speech(language)
#         st.success("✅ Sentence captured!")
#         st.write(f"Recognized {language} text: {text}")
import streamlit as st
import pyaudio #type:ignore
import json
from vosk import Model, KaldiRecognizer #type:ignore
from googletrans import Translator  # Import Google Translator
import threading

# Function to load the Vosk model based on the selected language
def load_model(language):
    if language == 'Telugu':
        return Model("model/vosk-model-small-te-0.42")
    elif language == 'Hindi':
        return Model("model/vosk-model-small-hi-0.22")
    elif language == 'gujarati':
        return Model("model/vosk-model-small-gu-0.42")
    elif language == 'English':
        return Model("model/vosk-model-small-en-in-0.4")
    else:
        raise ValueError("Unsupported language")

# Function to start listening and update the text variable
def listen_for_speech(language):
    model = load_model(language)
    rec = KaldiRecognizer(model, 16000)

    # Initialize PyAudio
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=8000)
    stream.start_stream()

    text = ""
    while True:
        data = stream.read(4000, exception_on_overflow=False)
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            text = result.get("text", "")
            if text:
                break  # Stop listening after capturing the first sentence

    stream.stop_stream()
    stream.close()
    p.terminate()
    return text

# Function to translate text to English using Google Translate
def translate_to_english(text, language):
    translator = Translator()
    if language == 'Telugu' or language == 'Hindi' or 'Gujarati' or 'English':
        translated = translator.translate(text, src=language, dest='en')
        return translated.text
    return text
# Function to diagnose UTI based on keywords in the text
def diagnose_uti(translated_text):
    # List of keywords related to urinary tract infection
    keywords = ["painful urination", "itching", "burning sensation while urination", "continuous urination"]
    
    # Check if any keyword is present in the translated text
    for keyword in keywords:
        if keyword.lower() in translated_text.lower():
            return "You have urinary tract infection"
    
    return "You are healthy"

# Streamlit UI
st.title("Multilingual Speech Recognition with Translation")
st.write("Click the button below to start listening and capture a sentence.")

# Language selection dropdown
language = st.selectbox("Select language", ("Telugu", "Hindi", "Gujarati","English"))

if st.button("Start Listening"):
    with st.spinner(f"Listening in {language}..."):
        text = listen_for_speech(language)
        st.success("✅ Sentence captured!")
        st.write(f"Recognized {language} text: {text}")
        
        # Translate to English
        translated_text = translate_to_english(text, language)
        st.write(f"Translated English text: {translated_text}")
        
        # Diagnose UTI based on the translated text
        diagnosis = diagnose_uti(translated_text)
        st.write(f"Diagnosis: {diagnosis}")
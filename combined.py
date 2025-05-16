import streamlit as st
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from gtts import gTTS #type:ignore
import tempfile
import joblib
import streamlit as st
import pyaudio #type:ignore
import json
from vosk import Model, KaldiRecognizer #type:ignore
from googletrans import Translator  # Import Google Translator
import threading

YES_NO_UTI =  {
    "English": ["Yes", "No"],
    "Hindi": ["हाँ", "नहीं"],
    "Bengali": ["হ্যাঁ", "না"],
    "Telugu": ["అవును", "కాదు"],
    "Tamil": ["ஆம்", "இல்லை"],
    "Kannada": ["ಹೌದು", "ಇಲ್ಲ"],
    "Malayalam": ["അതെ", "ഇല്ല"]
}
PREDICTION_TRANSLATIONS_UTI = {
    "UTI": {
        "English": "Urinary Tract Infection (UTI)",
        "Hindi": "यूरिनरी ट्रैक्ट संक्रमण (UTI)",
        "Bengali": "ইউরিনারি ট্র্যাক্ট ইনফেকশন (UTI)",
        "Telugu": "యూరినరీ ట్రాక్ ఇన్‌ఫెక్షన్ (UTI)",
        "Tamil": "மூத்திரதழை தொற்று (UTI)",
        "Kannada": "ಯೂರಿನರಿ ಟ್ರ್ಯಾಕ್ಟ್ ಸೋಂಕು (UTI)",
        "Malayalam": "യൂറിനറി ട്രാക്ക് ഇൻഫെക്ഷൻ (UTI)"
    },
    "No UTI": {
        "English": "No Urinary Tract Infection",
        "Hindi": "कोई यूरिनरी ट्रैक्ट संक्रमण नहीं",
        "Bengali": "কোনো ইউরিনারি ট্র্যাক্ট ইনফেকশন নেই",
        "Telugu": "యూరినరీ ట్రాక్ ఇన్‌ఫెక్షన్ లేదు",
        "Tamil": "யாதொரு மூத்திரதழை தொற்று இல்லை",
        "Kannada": "ಯಾವುದೇ ಯೂರಿನರಿ ಟ್ರ್ಯಾಕ್ಟ್ ಸೋಂಕು ಇಲ್ಲ",
        "Malayalam": "യൂറിനറി ട്രാക്ക് ഇൻഫെക്ഷൻ ഇല്ല"
    }
}
# Language mapping
LANG_TO_CODE = {
    "English": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Telugu": "te",
    "Tamil": "ta",
    "Kannada": "kn",
    "Malayalam": "ml"
}
languages_uti ={
    "English": [
        "Do you experience nausea?",
        "Do you have lumbar (lower back) pain?",
        "Do you feel a continuous urge to urinate?",
        "Do you experience pain during urination?",
        "Do you have burning, itching, or swelling at the urethral outlet?",
        "Do you have inflammation of the urinary bladder?",
        "Do you have nephritis of renal pelvis origin?"
    ],
    "Hindi": [
        "क्या आपको मतली आती है?",
        "क्या आपको कमर (पीठ के निचले हिस्से) में दर्द होता है?",
        "क्या आपको बार-बार पेशाब करने की इच्छा होती है?",
        "क्या आपको पेशाब करते समय दर्द होता है?",
        "क्या आपको मूत्रमार्ग के छिद्र में जलन, खुजली या सूजन होती है?",
        "क्या आपको मूत्राशय में सूजन है?",
        "क्या आपको गुर्दे की श्रोणि की सूजन (नेफ्राइटिस) है?"
    ],
    "Bengali": [
        "আপনি কি বমিভাব অনুভব করেন?",
        "আপনার কি লোয়ার ব্যাক (কোমরের) ব্যথা আছে?",
        "আপনার কি বারবার প্রস্রাবের তাগিদ অনুভূত হয়?",
        "আপনি কি প্রস্রাবের সময় ব্যথা অনুভব করেন?",
        "আপনার মূত্রনালীর প্রান্তে জ্বালাপোড়া, চুলকানি বা ফোলা আছে কি?",
        "আপনার কি মূত্রাশয়ে প্রদাহ হয়েছে?",
        "আপনার কি কিডনির পেলভিসের প্রদাহ (নেফ্রাইটিস) হয়েছে?"
    ],
    "Telugu": [
        "మీకు వాంతులుగా అనిపిస్తున్నాయా?",
        "మీకు నడుము (కిందటి వెన్నునొప్పి) నొప్పి ఉందా?",
        "మీకు తరచుగా మూత్ర విసర్జన చేయాలనే భావన ఉందా?",
        "మీరు మూత్ర విసర్జన సమయంలో నొప్పిని అనుభవిస్తున్నారా?",
        "మీరు మూత్రనాళం చివర వద్ద మండిపోవడం, గిరకటం లేదా వాపును అనుభవిస్తున్నారా?",
        "మీకు మూత్రాశయంలో వాపు ఉందా?",
        "మీకు మూత్రపిండం యొక్క పెల్విస్‌లో నెఫ్రైటిస్ ఉందా?"
    ],
    "Tamil": [
        "உங்களுக்கு வாந்தி உணர்வு ஏற்படுகிறதா?",
        "உங்களுக்கு முதுகின் கீழ் பகுதியில் வலி இருக்கிறதா?",
        "உங்களுக்கு அடிக்கடி சிறுநீர் கழிக்க வேண்டும் என்று தோன்றுகிறதா?",
        "நீங்கள் சிறுநீர் கழிக்கும் போது வலி ஏற்படுகிறதா?",
        "உங்கள் சிறுநீரக வெளிப்புறத்தில் எரிச்சல், அரிப்பு அல்லது வீக்கம் உள்ளதா?",
        "உங்களுக்கு சிறுநீர்ப்பை அழற்சி உள்ளதா?",
        "உங்களுக்கு சிறுநீரக அறையின் அழற்சி (நெஃப்ரைட்டிஸ்) உள்ளதா?"
    ],
    "Kannada": [
        "ನೀವು ವಾಂತಿ ಅನುಭವಿಸುತ್ತೀರಾ?",
        "ನೀವು ಕಡಿಮೆ ಬೆನ್ನುನೋವು ಅನುಭವಿಸುತ್ತೀರಾ?",
        "ನೀವು ನಿರಂತರವಾಗಿ ಮೂತ್ರ ವಿಸರ್ಜನೆ ಮಾಡುವ ಅಗತ್ಯವನ್ನು ಅನುಭವಿಸುತ್ತೀರಾ?",
        "ನೀವು ಮೂತ್ರ ವಿಸರ್ಜನೆ ಮಾಡುವಾಗ ನೋವು ಅನುಭವಿಸುತ್ತೀರಾ?",
        "ನೀವು ಮೂತ್ರನಾಳದ ಹೊರಭಾಗದಲ್ಲಿ ಉರಿದಂತಹ, ಉರಿಯೂತ ಅಥವಾ ಊತ ಅನುಭವಿಸುತ್ತೀರಾ?",
        "ನೀವು ಮೂತ್ರಪಿಂಡದ ಆವರಣದಲ್ಲಿ ಉರಿಯೂತ ಅನುಭವಿಸುತ್ತೀರಾ?",
        "ನೀವು ಮೂತ್ರಪಿಂಡದ ಮೂತ್ರಪೆಲ್ವಿಸ್ ನೆಫ್ರೈಟಿಸ್ ಹೊಂದಿದ್ದೀರಾ?"
    ],
    "Malayalam": [
        "നിങ്ങൾക്ക് വാന്തിയുണ്ടോ?",
        "നിങ്ങൾക്ക് താഴത്തെ പിൻഭാഗത്തിൽ വേദനയുണ്ടോ?",
        "നിങ്ങൾക്ക് തുടർച്ചയായി മൂത്രമൊഴിക്കേണ്ടതാണെന്ന് തോന്നുന്നുണ്ടോ?",
        "നിങ്ങൾക്ക് मूത്രमൊഴിക്കുമ്പോൾ വേദനയുണ്ടോ?",
        "നിങ്ങൾക്ക് मूത്രനാളി പുറത്ത് കത്തുന്നത്, കടിച്ചു കൊണ്ട് ഉള്ളത്, അല്ലെങ്കിൽ വീക്കം ഉണ്ടോ?",
        "നിങ്ങൾക്ക് मूത്രാശയത്തിലുണ്ടാകുന്ന അണുബാധയുണ്ടോ?",
        "നിങ്ങൾക്ക് വൃക്കയുടെ मूത്രपെൽवിസിൽ നെഫ്രൈറ്റിസ് ഉണ്ടോ?"
    ]
}

languages_pcos = {
    "English": {
        "questions": [
            "Do you have pelvic pain?", "Are your periods irregular?", "Do you urinate frequently?",
            "Do you experience a burning sensation while urinating?", "Have you been diagnosed with ovarian cysts?",
            "Are you facing infertility issues?", "Do you have lower back pain?", "Do you feel nauseous often?",
            "Do you experience frequent fatigue?", "Do you have persistent headaches?", "Do you suffer from abdominal pain?",
            "Have you experienced unexplained weight gain?", "Are you experiencing hair loss?", "Do you have severe acne issues?",
            "Do you have hormonal imbalance?", "Do you experience frequent mood swings?"
        ],
        "options": ["Yes", "No"],
        "language_code": "en"
    },
    "Hindi": {
        "questions": [
            "क्या आपको श्रोणि में दर्द है?", "क्या आपके पीरियड्स अनियमित हैं?", "क्या आपको बार-बार पेशाब आता है?",
            "क्या आपको पेशाब करते समय जलन होती है?", "क्या आपको अंडाशय की सिस्ट है?",
            "क्या आपको बांझपन की समस्या है?", "क्या आपको पीठ के निचले हिस्से में दर्द है?", "क्या आपको अक्सर मतली आती है?",
            "क्या आपको बार-बार थकान होती है?", "क्या आपको लगातार सिरदर्द होता है?", "क्या आपको पेट दर्द होता है?",
            "क्या आपका वजन बिना किसी कारण बढ़ रहा है?", "क्या आपके बाल झड़ रहे हैं?", "क्या आपको गंभीर मुँहासे की समस्या है?",
            "क्या आपके हार्मोन असंतुलित हैं?", "क्या आपका मूड बार-बार बदलता रहता है?"
        ],
        "options": ["हाँ", "नहीं"],
        "language_code": "hi"
    },
    "Bengali": {
        "questions": [
            "আপনার কি পেলভিক ব্যথা আছে?", "আপনার কি অনিয়মিত মাসিক হয়?", "আপনার কি প্রায়শই প্রস্রাব হয়?",
            "আপনার প্রস্রাব করার সময় কি জ্বালাপোড়া হয়?", "আপনার কি ডিম্বাশয়ের সিস্ট হয়েছে?",
            "আপনি কি বন্ধ্যাত্বের সম্মুখীন?", "আপনার কি নিম্ন পিঠে ব্যথা আছে?", "আপনি কি প্রায়ই বমি বমি ভাব অনুভব করেন?",
            "আপনার কি ঘন ঘন ক্লান্তি হয়?", "আপনার কি বারবার মাথাব্যথা হয়?", "আপনার কি পেটে ব্যথা হয়?",
            "আপনার ওজন কি অপ্রত্যাশিতভাবে বেড়ে গেছে?", "আপনার কি চুল পড়ছে?", "আপনার কি গুরুতর ব্রণ সমস্যা আছে?",
            "আপনার হরমোন কি ভারসাম্যহীন?", "আপনার কি প্রায়ই মুড পরিবর্তন হয়?"
        ],
        "options": ["হ্যাঁ", "না"],
        "language_code": "bn"
    }
,
    "Telugu":{ "questions": [
        "మీకు పెల్విక్ నొప్పి ఉందా?", "మీ పీరియడ్లు అసాధారణంగా ఉన్నాయా?", "మీరు తరచుగా మూత్ర విసర్జన చేయాలనుకుంటున్నారా?",
        "మీరు మూత్ర విసర్జన సమయంలో మంట అనుభవిస్తున్నారా?", "మీరు ఓవేరియన్ సిస్టుతో ఉన్నారా?",
        "మీకు సంతానలేమి సమస్య ఉందా?", "మీరు తక్కువ వెన్నునొప్పి అనుభవిస్తున్నారా?", "మీరు తరచుగా వికారం అనుభవిస్తున్నారా?",
        "మీరు తరచుగా అలసట అనుభవిస్తున్నారా?", "మీకు తలనొప్పి తరచుగా వస్తుందా?", "మీరు కడుపునొప్పి అనుభవిస్తున్నారా?",
        "మీ బరువు అనూహ్యంగా పెరిగిందా?", "మీ జుట్టు ఊడిపోతుందా?", "మీరు తీవ్రమైన మొటిమల సమస్యను అనుభవిస్తున్నారా?",
        "మీ హార్మోన్లు అసమతుల్యతలో ఉన్నాయా?", "మీ మూడ్ తరచుగా మారుతుందా?"
    ],
    "options": ["అవును", "కాదు"],
     "language_code": "te"
    },
    "Tamil": { "questions":[
        "உங்களுக்கு பெல்விக் வலி இருக்கிறதா?", "உங்கள் மாதவிடாய் முறையற்றதா?", "நீங்கள் அடிக்கடி சிறுநீர் கழிக்க வேண்டுமா?",
        "நீங்கள் சிறுநீர் கழிக்கும் போது எரிச்சல் உணருகிறீர்களா?", "உங்களுக்கு முட்டை அணை சிச்ட் உள்ளது?",
        "நீங்கள் கருத்தரிக்க முடியாத பிரச்சனை கொண்டுள்ளீர்களா?", "உங்களுக்கு கீழ் முதுகு வலி இருக்கிறதா?", "நீங்கள் அடிக்கடி வாந்தி உணர்கிறீர்களா?",
        "நீங்கள் அடிக்கடி சோர்வு உணர்கிறீர்களா?", "உங்களுக்கு தொடர்ந்து தலைவலி உள்ளதா?", "நீங்கள் வயிற்று வலியால் பாதிக்கப்படுகிறீர்களா?",
        "உங்கள் எடை எதிர்பாராத விதமாக அதிகரித்ததா?", "உங்கள் முடி உதிர்வதா?", "நீங்கள் கடுமையான முகப்பரு பிரச்சனையை அனுபவிக்கிறீர்களா?",
        "உங்கள் ஹார்மோன்கள் சமநிலையில் இல்லையா?", "உங்கள் மனநிலை அடிக்கடி மாறுகிறதா?"
    ],
    "options": ["ஆம்", "இல்லை"],
     "language_code": "ta"
    },
    "Malayalam": {
         "questions":[
        "നിങ്ങൾക്ക് പെൽവിക് വേദനയുണ്ടോ?", "നിങ്ങളുടെ കാലയളവ് അക്രമശീലമാണോ?", "നിങ്ങൾക്ക് ആവർത്തിച്ചുള്ള മൂത്രസ്രാവമാണോ?",
        "നിങ്ങൾക്ക് മൂത്രസ്രാവത്തിനിടെ കത്തുന്നതാണോ?", "നിങ്ങൾക്ക് ഓവേറിയൻ സിസ്റ്റ് ഉണ്ടോ?",
        "നിങ്ങൾക്ക് വന്ധ്യത പ്രശ്നമുണ്ടോ?", "നിങ്ങൾക്ക് താഴത്തെ കഴുത്തുവേദന ഉണ്ടോ?", "നിങ്ങൾക്ക് ആവർത്തിച്ചുള്ള ഛർദി അനുഭവപ്പെടുമോ?",
        "നിങ്ങൾക്ക് ആവർത്തിച്ചുള്ള ക്ഷീണം അനുഭവപ്പെടുമോ?", "നിങ്ങൾക്ക് സ്ഥിരമായ തലവേദന ഉണ്ടോ?", "നിങ്ങൾക്ക് വയറുവേദന അനുഭവപ്പെടുമോ?",
        "നിങ്ങളുടെ ഭാരം അനിയന്ത്രിതമായി കൂടിയിട്ടുണ്ടോ?", "നിങ്ങളുടെ മുടി പതിക്കുന്നു?", "നിങ്ങൾ ഗുരുതരമായ മുന്പുള്ള പ്രശ്നം അനുഭവപ്പെടുന്നുണ്ടോ?",
        "നിങ്ങളുടെ ഹോർമോണുകൾ അസന്തുലിതമാണോ?", "നിങ്ങളുടെ മനോഭാവം ആവർത്തിച്ചും മാറുന്നുണ്ടോ?"
    ],
    "options": ["അതെ", "അല്ല"],
     "language_code": "ml"
},
    "Kannada": {
        "questions": [
            "ನಿಮಗೆ ಪೆಲ್ವಿಕ್ ನೋವು ಇದೆಯಾ?", "ನಿಮ್ಮ ಮುಂಗಡ ಸಮಯ ಅಸ್ವಾಭಾವಿಕವಾಗಿದೆಯಾ?", "ನೀವು ಪದೇ ಪದೇ ಮೂತ್ರ ವಿಸರ್ಜನೆ ಮಾಡಬೇಕಾ?",
            "ನೀವು ಮೂತ್ರ ವಿಸರ್ಜನೆ ಮಾಡುವಾಗ ಸುಡುವುದು ಅನುಭವಿಸುತ್ತೀರಾ?", "ನಿಮಗೆ ಓವೇರಿಯನ್ ಸಿಸ್ಟ್ ಇದೆಯಾ?",
            "ನೀವು ಸಂತಾನ ಹೀನತೆ ಸಮಸ್ಯೆ ಎದುರಿಸುತ್ತೀರಾ?", "ನಿಮಗೆ ಕೆಳಗಿನ ಬೆನ್ನು ನೋವು ಇದೆಯಾ?", "ನೀವು ಅಸ್ವಸ್ಥರಾಗಿದ್ದೀರಾ?",
            "ನೀವು ನಿಯಮಿತವಾಗಿ ಕಳೆದುಕೊಳ್ಳುತ್ತಿದ್ದೀರಾ?", "ನಿಮಗೆ ನಿರಂತರ ತಲೆನೋವು ಇದೆಯಾ?", "ನೀವು ಹೊಟ್ಟೆ ನೋವು ಅನುಭವಿಸುತ್ತೀರಾ?",
            "ನಿಮ್ಮ ತೂಕ ನಿರೀಕ್ಷಿತವಾಗಿ ಹೆಚ್ಚಿದೆಯಾ?", "ನಿಮ್ಮ ಕೂದಲು ಬೀಳುತ್ತಿದೆಯಾ?", "ನೀವು ತೀವ್ರವಾದ ಮೊಡವೆ ಸಮಸ್ಯೆಯನ್ನು ಅನುಭವಿಸುತ್ತೀರಾ?",
            "ನಿಮ್ಮ ಹಾರ್ಮೋನುಗಳು ಅಸ್ತವ್ಯಸ್ತವಾಗಿದೆಯಾ?", "ನಿಮ್ಮ ಮನೋಭಾವವು ಪದೇ ಪದೇ ಬದಲಾಯಿಸುತ್ತಿದೆಯಾ?"
        ],
        "options": ["ಹೌದು", "ಇಲ್ಲ"],
         "language_code": "kn"
    }

}
def load_model(language):
    if language == 'Telugu':
        return Model("model/vosk-model-small-te-0.42")
    elif language == 'Hindi':
        return Model("model/vosk-model-small-hi-0.22")
    elif language == 'Gujarati':
        return Model("model/vosk-model-small-gu-0.42")
    elif language == 'English':
        return Model("model/vosk-model-small-en-in-0.4")
    else:
        raise ValueError("Unsupported language")

import time

def list_input_devices():
    p = pyaudio.PyAudio()
    device_list = []

    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            device_list.append({"index": i, "name": info["name"]})
    
    p.terminate()
    return device_list

def listen_for_speech(language, device_index=None, listen_duration=10):
    model = load_model(language)
    rec = KaldiRecognizer(model, 16000)
    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=8000)
    stream.start_stream()

    collected_texts = []
    start_time = time.time()

    while time.time() - start_time < listen_duration:
        data = stream.read(4000, exception_on_overflow=False)
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            recognized = result.get("text", "")
            if recognized:
                collected_texts.append(recognized)

    final_result = json.loads(rec.FinalResult())
    final_text = final_result.get("text", "")
    if final_text:
        collected_texts.append(final_text)

    stream.stop_stream()
    stream.close()
    p.terminate()

    full_text = " ".join(collected_texts).strip()
    return full_text


def translate_to_english(text, language):
    translator = Translator()
    if language in LANG_TO_CODE:
        translated = translator.translate(text, src=LANG_TO_CODE[language], dest='en')
        return translated.text
    return text

def diagnose_uti(translated_text):
    
    keywords = ["painful urination", "itching", "burning sensation while urination", "continuous urination"]


    for keyword in keywords:
        if keyword.lower() in translated_text.lower():
            return "You have urinary tract infection"

    return "You are healthy"

def audioo():
    st.title("🎙️ Multilingual Speech Recognition with Translation")
    st.write("Select your language and microphone, then click the button to start listening.")

    language = st.selectbox("Select language", ("Telugu", "Hindi", "Gujarati", "English"))

   
    devices = list_input_devices()
    device_names = [f"{dev['index']}: {dev['name']}" for dev in devices]
    selected_device = st.selectbox("Select audio input device", device_names)

   
    selected_device_index = int(selected_device.split(":")[0]) if selected_device else None

    if st.button("🎤 Start Listening"):
        with st.spinner(f"Listening in {language} using device {selected_device}..."):
            text = listen_for_speech(language, device_index=selected_device_index)

            if not text.strip():
                st.warning("⚠️ No speech was detected. Please try again.")
                return

            st.success("✅ Sentence captured!")
            st.write(f"Recognized {language} text: {text}")

            translated_text = translate_to_english(text, language)
            st.write(f"Translated English text: {translated_text}")

            diagnosis = diagnose_uti(translated_text)
            st.write(f"Diagnosis: {diagnosis}")

def play_audio(text, language_code):
    tts = gTTS(text=text, lang=language_code, slow=False)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        return fp.name

def play_audio(text, language_code):
    tts = gTTS(text=text, lang=language_code, slow=False)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        return fp.name

# Function to train the PCOS model
def wrangle_and_train_pcos():
    if not os.path.exists("gddes_pcos.pkl"):
        df = pd.read_csv(r"C:\Users\cdhan\Downloads\filtered_pcos_healthy.csv")
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        y = y.map({"PCOS": 1, "Healthy": 0})
        encoder = OneHotEncoder(handle_unknown='ignore')
        X_encoded = encoder.fit_transform(X).toarray()
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
        model = SVC(kernel='linear', probability=True)
        model.fit(X_train, y_train)
        with open("gddes_pcos.pkl", "wb") as f:
            joblib.dump((model, encoder), f)

# Function to train the UTI model
def wrangle_and_train_uti():
    model_path = "C:/Users/cdhan/Downloads/project/uti_model.pkl"  # Update path if necessary
    if not os.path.exists(model_path):  # Check if the model exists
        print("Training and saving the model...")
        df = pd.read_csv("C:/Users/cdhan/Downloads/uti_final.csv")
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        encoder = OneHotEncoder(handle_unknown='ignore')
        X_encoded = encoder.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

        model = SVC(kernel='rbf', probability=True)
        model.fit(X_train, y_train)

        # Save model and encoder
        with open(model_path, "wb") as f:
            pickle.dump((model, encoder), f)  # Save both model and encoder
        print(f"Model saved at {model_path}")
    else:
        print(f"Model already exists at {model_path}, no training needed.")


def pcos_diagnosis():
    wrangle_and_train_pcos()
    st.title("PCOS Diagnosis System")
    language = st.selectbox("Select Language", list(languages_pcos.keys()))
    questions = languages_pcos[language]["questions"]
    options = languages_pcos[language]["options"]
    language_code = languages_pcos[language]["language_code"]
    responses = []
    for i, question in enumerate(questions):
        response = st.radio(question, options, key=f"pcos_q{i}")
        responses.append(1 if response == options[0] else 0)
        if st.button(f"🎧 Play Question {i+1}", key=f"play_pcos_{i}"):
            audio_file = play_audio(question, language_code)
            st.audio(audio_file, format="audio/mp3")
    if st.button("Submit"):
        with open("gddes_pcos.pkl", "rb") as f:
            model, encoder = joblib.load(f)
        input_encoded = encoder.transform([responses]).toarray()
        prediction = model.predict(input_encoded)[0]
        prediction_label = "PCOS" if prediction == 1 else "Healthy"
        st.success(f"Predicted Diagnosis: {prediction_label}")

def uti_diagnosis():

    wrangle_and_train_uti()
    st.title("UTI Diagnosis System")

    selected_language = st.selectbox("🌐 Select Language", list(LANG_TO_CODE.keys()))
    questions = languages_uti[selected_language]
    options = YES_NO_UTI[selected_language]
    yes_no_map = dict(zip(options, ["yes", "no"]))
    
    responses = []  


    for i, question in enumerate(questions):
        st.write(question)
        response = st.radio(f"Question {i+1}", options, key=f"uti_q{i}")
        responses.append(yes_no_map[response])


        if st.button(f"🎧 Play Question {i+1}", key=f"play_uti_{i}"):
            audio_file = play_audio(question, LANG_TO_CODE[selected_language])
            st.audio(audio_file, format="audio/mp3")

    # Submit for prediction
    if st.button("Submit Questionnaire"):
        try:
            
            with open("C:/Users/cdhan/Downloads/project/uti_model.pkl", "rb") as f:
                model, encoder = pickle.load(f)


            
            X_input = encoder.transform([responses])  

            
            prediction = model.predict(X_input)

            translated_result = PREDICTION_TRANSLATIONS_UTI["UTI" if prediction[0] == 1 else "No UTI"][selected_language]
            st.success(f"Prediction Result: {translated_result}")

        except ValueError as e:
            st.error(f"Error during prediction: {e}")
            st.error("Please ensure all inputs are valid and match the expected format.")
        except FileNotFoundError as e:
            st.error(f"Model file not found at the specified path: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
# Navigator
def main():
    st.sidebar.title("🔍 Diagnosis Navigator")
    choice = st.sidebar.radio("Choose a Diagnosis System", ["PCOS Diagnosis", "UTI Diagnosis","Audio"])
    if choice == "PCOS Diagnosis":
        pcos_diagnosis()
    elif choice == "UTI Diagnosis":
        uti_diagnosis()
    elif choice == "Audio":
        audioo()
if __name__ == "__main__":
    main()

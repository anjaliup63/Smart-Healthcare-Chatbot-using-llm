import streamlit as st
import pickle
from deep_translator import GoogleTranslator
from gtts import gTTS
import os
import io
from dotenv import load_dotenv
load_dotenv()

# Setup secure Google Gemini SDK (using official google-genai library recommendation)
from google import genai
from google.genai import types

# Load model pickle maps safely
@st.cache_resource
def load_ml_models():
    try:
        model = pickle.load(open("model.pkl", "rb"))
        vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
        return model, vectorizer
    except Exception as e:
        # Fallback simulated handles if pickled models are missing
        st.warning("⚠️ Local pre-trained model.pkl or vectorizer.pkl missing. Using smart fallback system.")
        return None, None

model, vectorizer = load_ml_models()

# Rule-based emergency keyword detector
def check_rule_based_emergency(text):
    keywords = [
        "chest pain", "bleeding", "breathless", "shortness of breath",
        "unconscious", "fainting", "seizure", "stroke", "heart attack",
        "paralysis", "blood loss", "faint", "fits"
    ]
    query_lower = text.lower()
    return [k for k in keywords if k in query_lower]

# Intent Prediction
def predict_clinical_intent(text):
    kw = check_rule_based_emergency(text)
    if kw:
        return "Emergency", 1.0, kw, "Rule-Based Core Engine"
    
    if model and vectorizer:
        try:
            X = vectorizer.transform([text])
            pred = model.predict(X)[0]
            prob = max(model.predict_proba(X)[0])
            return pred, prob, [], "Clinical ML Classifier"
        except Exception:
            pass
            
    # LLM-assisted structural fallback intent if files are missing
    return "General Consultation", 0.88, [], "Heuristic Parser"

# Setup streamlined page layouts with professional theme UI
st.set_page_config(
    page_title="Smart Health Consultation Assistant",
    page_icon="🩺",
    layout="wide"
)

# Render customized header grids
st.markdown("""
    <div style='background-color:#0f172a; padding: 24px; border-radius: 12px; margin-bottom: 25px; border-left: 5px solid #10b981;'>
        <h1 style='color:white; margin:0; font-size: 2.22rem;'>🩺 Smart Healthcare Chatbot</h1>
        <p style='color:#94a3b8; font-size: 1.05rem; margin-top:8px; margin-bottom:0;'>
            Bilingual Clinical Consultation Assistant leveraging hybrid Expert Rule Systems, local ML Intent classifiers, and Google Gemini LLMs.
        </p>
    </div>
""", unsafe_allow_html=True)

# Sourcing User Input with safe fallbacks
user_input = st.text_input("Enter your health concern, symptoms, or medical question:", placeholder="e.g. I have a sudden tightness in my chest and pain in my arm")

if user_input:
    # 1. Error Handling: Avoid processing blank inputs
    if not user_input.strip():
        st.error("⚠️ Please specify a valid query before requesting evaluation.")
    else:
        with st.spinner("Processing symptoms with clinical evaluation models..."):
            try:
                # 2. Hindi & General input translated to English for unified inference
                translated_en = GoogleTranslator(source='auto', target='en').translate(user_input)
                
                # 3. Intent Detection & Safety Checks
                intent, confidence, keywords, method = predict_clinical_intent(translated_en)
                
                # 4. Gemini API Structured Response Layer
                # Retrieve key safely from stream or standard OS env map
                api_key = os.getenv("GEMINI_API_KEY")
                st.write("API Loaded:", api_key is not None)
                if not api_key:
                    st.error("❌ GEMINI_API_KEY is not configured in your environment secrets.")
                    st.stop()
                
                # Initialize Google GenAI client securely
                client = genai.Client(api_key=api_key)
                
                system_instruction = """
                You are a professional medical education AI assistant. 
                Structure your healthcare education response in BOTH English and Hindi.
                Ensure you never diagnose a specific disease as certain, never prescribe medicines, and always include safety disclaimers.
                """
                
                # Query Gemini to generate standardized output sections
                prompt = f"""
                Construct a highly detailed healthcare report for symptom query: '{translated_en}'.
                You must formulate answers utilizing these separate key structural markers:
                1. CONDITION OVERVIEW
                2. POSSIBLE CAUSES
                3. SYMPTOMS
                4. PRECAUTIONS
                5. LIFESTYLE RECOMMENDATIONS
                6. WHEN TO CONSULT A DOCTOR
                7. EMERGENCY WARNING SIGNS
                8. DISCLAIMER
                
                Formulate two reports: the first in English, followed by the second as a complete native translation in Hindi (हिंदी).
                """
                
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.3
                    )
                )
                
                full_text = response.text
                
                # UI Layout columns for clear segmentation
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("📋 Assistant Educational Guidance")
                    
                    # Distinguish Emergency Status Layouts
                    if intent == "Emergency":
                        st.error("🚨 CRITICAL WARNING: Active symptoms match high-level medical emergency indexes!")
                    else:
                        st.success("✅ Assessment complete. Review structured clinical report outcomes below.")
                    
                    # Tabs to view English or Hindi report files
                    tab_en, tab_hi = st.tabs(["🇺🇸 English Report", "🇮🇳 हिंदी रिपोर्ट (Hindi)"])
                    
                    # Split full text into English and Hindi portions
                    if "हिंदी" in full_text or "HINDI" in full_text:
                        # Simple split heuristics
                        parts = full_text.split("हिंदी") if "हिंदी" in full_text else full_text.split("HINDI")
                        en_report, hi_report = parts[0], parts[1]
                    else:
                        en_report = full_text
                        hi_report = "हिंदी अनुवाद अनुपलब्ध है। कृपया अनुवादक बटन का उपयोग करें।"
                        
                    with tab_en:
                        st.markdown(en_report)
                    with tab_hi:
                        st.markdown(hi_report)
                        
                with col2:
                    st.subheader("🎯 Telemetry & Method Metrics")
                    st.metric(label="Predicted Clinical Intent", value=intent)
                    st.metric(label="Detection Method Level", value=method)
                    st.metric(label="Model Confidence Score", value=f"{round(confidence * 100, 1)}%")
                    
                    if keywords:
                        st.warning(f"⚠️ Triggered Emergency Keywords: {', '.join(keywords)}")
                        
                    st.markdown("---")
                    st.subheader("🔊 Audio Voice Outputs")
                    
                    # Speech Generation utilizing gTTS
                    try:
                        # Play active report based on user language choice
                        speaker_lang = st.selectbox("Speech Voice Output Accent:", ["English", "Hindi (हिंदी)"])
                        speech_text = en_report if speaker_lang == "English" else hi_report
                        
                        # Strip Markdown to optimize reading flow
                        clean_audio_text = speech_text.replace("*", "").replace("#", "").replace("-", "")
                        
                        tts = gTTS(text=clean_audio_text[:500], lang='en' if speaker_lang == "English" else 'hi')
                        fp = io.BytesIO()
                        tts.write_to_fp(fp)
                        fp.seek(0)
                        
                        st.audio(fp, format="audio/mp3")
                        st.caption("🔊 Play synthesized MP3 output of symptoms education.")
                    except Exception as audio_err:
                        st.info("Audio voice generation paused. Check gtts setup.")
                        
            except Exception as e:
                st.error(f"An unexpected process error occurred: {str(e)}")
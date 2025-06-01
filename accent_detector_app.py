
import streamlit as st
import requests
import tempfile
import os
import moviepy.editor as mp
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier

# --------- THEME TOGGLE -----------
if "theme" not in st.session_state:
    st.session_state.theme = "light"

with st.sidebar:
    st.markdown("## 🎨 Theme")
    mode = st.radio("Choose app mode", ["Light", "Dark"], index=0 if st.session_state.theme == "light" else 1)
    st.session_state.theme = "light" if mode == "Light" else "dark"
    st.markdown("---")
    st.write("Accent Detector by [Your Name]")

# --------- CSS FOR MODERN UI --------
def get_css(theme):
    if theme == "dark":
        bg = "#222b36"
        panel = "#222a34"
        text = "#e2eaf2"
        card = "#20232a"
        sep = "#3a4453"
        btn = "#184d7a"
        result_shadow = "0 2px 18px 0 rgba(0,0,0,0.18)"
    else:
        bg = "#f7f8fa"
        panel = "#fff"
        text = "#22314a"
        card = "#fff"
        sep = "#babfc7"
        btn = "#1a8cff"
        result_shadow = "0 2px 18px 0 rgba(25,80,80,0.10)"
    css = f"""
    <style>
    html, body, [class*="css"]  {{
        font-family: 'Inter', 'Segoe UI', sans-serif;
        background-color: {bg} !important;
        color: {text} !important;
    }}
    .header-title {{
        font-size:2.3em;
        font-weight:700;
        letter-spacing:-1.5px;
        margin-top: 0.3em;
        margin-bottom:0.1em;
        color: {text};
    }}
    .subtitle {{
        font-size:1.07em;
        color:#5f6c7c;
        margin-bottom:2.5em;
        margin-top: 0.5em;
        font-weight: 500;
        letter-spacing:-0.15px;
    }}
    .duo-panel {{
        display: flex;
        justify-content: center;
        align-items: start;
        gap: 2em;
        margin-bottom: 2em;
        flex-wrap: wrap;
    }}
    .panel-card {{
        background: {panel};
        border-radius: 1em;
        box-shadow: 0 1px 12px 0 rgba(50,70,100,0.08);
        padding: 2.1em 2em 1.4em 2em;
        min-width: 310px;
        max-width: 360px;
        width: 100%;
        color: {text};
        border: 1px solid {sep};
    }}
    .or-sep {{
        font-weight:600;
        color:{sep};
        font-size:1.12em;
        padding: 1.1em 0 0.8em 0;
        text-align:center;
    }}
    .stButton>button {{
        width: 100%;
        padding: 0.65em 0;
        font-size: 1.17em;
        font-weight: 700;
        border-radius: 8px;
        background: linear-gradient(90deg, {btn} 0%, #37c8ab 100%);
        color: white;
        border: none;
        margin-top:1em;
        transition: box-shadow 0.18s;
    }}
    .stButton>button:hover {{
        box-shadow: 0 1px 8px 0 #1a8cff33;
        background: linear-gradient(90deg, {btn} 15%, #27cbaa 85%);
    }}
    .result-card {{
        background: {card};
        border-radius: 1.2em;
        padding: 2.3em 2em 2em 2em;
        box-shadow: {result_shadow};
        max-width: 410px;
        margin: 2.3em auto 2em auto;
        text-align: center;
        font-size:1.19em;
        border: 1px solid {sep};
        color: {text};
    }}
    .result-accent {{
        font-size:1.7em;
        font-weight:700;
        letter-spacing: -1px;
        color:#1a8cff;
        margin-bottom: 0.2em;
    }}
    .result-confidence {{
        font-size:1.13em;
        color:#15b374;
        font-weight:600;
        margin-top: 0.6em;
    }}
    @media (max-width: 720px) {{
        .duo-panel {{
            flex-direction: column;
            gap: 0.5em;
        }}
        .panel-card {{
            max-width: 100%;
            padding: 1.3em 1em 1em 1em;
        }}
        .result-card {{
            padding: 1.2em 0.5em 1.2em 0.5em;
        }}
    }}
    </style>
    """
    return css

st.markdown(get_css(st.session_state.theme), unsafe_allow_html=True)

# --- Accent Model Loader ---
@st.cache_resource
def load_accent_model():
    model = EncoderClassifier.from_hparams(
        source="Jzuluaga/accent-id-commonaccent_xlsr-en-english",
        savedir="pretrained_models/accent-id"
    )
    return model

def download_video(video_url):
    try:
        video_data = requests.get(video_url, stream=True)
        if video_data.status_code == 200:
            tmp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            for chunk in video_data.iter_content(chunk_size=1024*1024):
                if chunk:
                    tmp_video_file.write(chunk)
            tmp_video_file.close()
            return tmp_video_file.name
        else:
            return None
    except Exception as e:
        st.error(f"Download error: {e}")
        return None

def save_uploaded_file(uploaded_file):
    try:
        suffix = os.path.splitext(uploaded_file.name)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            return tmp_file.name
    except Exception as e:
        st.error(f"Upload error: {e}")
        return None

def extract_audio(video_path):
    try:
        video_clip = mp.VideoFileClip(video_path)
        audio_path = video_path.replace('.mp4', '.wav')
        video_clip.audio.write_audiofile(audio_path, codec='pcm_s16le')
        return audio_path
    except Exception as e:
        st.error(f"Audio extraction error: {e}")
        return None

def classify_accent(audio_path, model):
    try:
        signal, fs = torchaudio.load(audio_path)
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        if fs != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
            signal = resampler(signal)
        prediction = model.classify_batch(signal)
        probs = torch.nn.functional.softmax(prediction[1], dim=1)[0]
        top_idx = torch.argmax(probs).item()
        label = model.hparams.label_encoder.decode_torch(torch.tensor([top_idx]))[0]
        confidence = probs[top_idx].item()
        return label, confidence
    except Exception as e:
        st.error(f"Classification error: {e}")
        return None, None

# ---- Main UI ----
st.markdown('<div style="text-align:center;"><span style="font-size:2.3em;">🗣️</span></div>', unsafe_allow_html=True)
st.markdown('<div class="header-title" style="text-align:center;">Accent Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle" style="text-align:center;">Identify English accents in any video or audio.<br>Upload a file or paste a link. Light & dark modes.</div>', unsafe_allow_html=True)

# --- Duo Panel Input ---
st.markdown('<div class="duo-panel">', unsafe_allow_html=True)

# --- URL Panel ---
st.markdown('<div class="panel-card">', unsafe_allow_html=True)
st.markdown('<b>Paste a Video or Audio URL</b>', unsafe_allow_html=True)
video_url = st.text_input("", placeholder="https://... (MP4, WAV, Loom, etc.)")
st.markdown('</div>', unsafe_allow_html=True)

# --- OR separator ---
st.markdown('<div class="or-sep">or</div>', unsafe_allow_html=True)

# --- Upload Panel ---
st.markdown('<div class="panel-card">', unsafe_allow_html=True)
st.markdown('<b>Upload a File</b>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["mp4", "mov", "avi", "mkv", "webm", "wav", "mp3", "ogg"])
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

analyze = st.button("✨ Analyze Accent")

input_video_path = None

if analyze:
    if uploaded_file is not None:
        input_video_path = save_uploaded_file(uploaded_file)
    elif video_url:
        input_video_path = download_video(video_url)
    else:
        st.warning("Please upload a file or paste a video/audio URL.")

    if input_video_path:
        with st.spinner("Analyzing... This might take up to 1 minute."):
            audio_path = extract_audio(input_video_path)
            if audio_path:
                model = load_accent_model()
                label, confidence = classify_accent(audio_path, model)
                os.remove(audio_path)
                os.remove(input_video_path)
            else:
                label = None
                confidence = None

        if label:
            st.markdown(
                f'<div class="result-card">'
                f'<div class="result-accent">{label.capitalize()}</div>'
                f'<div class="result-confidence">Confidence: {round(confidence*100,2)}%</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        elif label is None:
            st.warning("Could not detect an accent. Try another video or check your link.")

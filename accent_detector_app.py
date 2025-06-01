
import streamlit as st
import tempfile
import requests
import moviepy.editor as mp
import openai
import os

# --- Configuration ---
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Accent Detector", layout="centered")

st.title("🎙️ English Accent Detector")
st.markdown("Upload a video file or paste a public video URL to detect the speaker's accent.")

# -- File upload or URL input
option = st.radio("Choose input method:", ["Upload video file", "Paste video URL"])

video_file = None
if option == "Upload video file":
    video_file = st.file_uploader("Upload your video file", type=["mp4", "mov", "webm"])
else:
    video_url = st.text_input("Enter video URL (direct link to .mp4)")
    if video_url:
        response = requests.get(video_url)
        if response.status_code == 200:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(response.content)
            video_file = tmp.name

if video_file and st.button("Analyze"):
    with st.spinner("Extracting audio..."):
        try:
            if isinstance(video_file, str):
                video_path = video_file
            else:
                # Save uploaded file to temp file
                temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                temp_video_path.write(video_file.read())
                temp_video_path.close()
                video_path = temp_video_path.name

            # Extract audio
            video = mp.VideoFileClip(video_path)
            audio_path = video_path.replace(".mp4", ".wav")
            video.audio.write_audiofile(audio_path)

            # Transcribe using OpenAI Whisper
            with open(audio_path, "rb") as audio_file:
                st.info("Transcribing audio with Whisper...")
                transcript_response = openai.Audio.transcribe("whisper-1", audio_file)

            transcription = transcript_response["text"]
            st.success("Transcription complete!")

            # Analyze accent with GPT
            st.info("Analyzing accent with GPT-4...")
            prompt = f"Analyze the following transcription and determine the speaker's English accent (e.g., British, American, Australian). Provide a confidence score between 0 and 100%, and a short explanation.\n\nTranscription:\n{transcription}"
            chat_response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )

            gpt_text = chat_response.choices[0].message.content
            st.markdown("### 🧠 Accent Analysis")
            st.markdown(gpt_text)

        except Exception as e:
            st.error(f"An error occurred: {e}")

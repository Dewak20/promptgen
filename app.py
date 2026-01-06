import streamlit as st
import google.generativeai as genai
from openai import OpenAI
import tempfile
import time
import os
import cv2
import base64
import yt_dlp
import re  # Penting untuk pembersihan teks

# --- FUNGSI BANTUAN ---
def download_video_from_url(url):
    try:
        ydl_opts = {'format': 'best[ext=mp4]/best', 'outtmpl': 'temp_video_%(id)s.%(ext)s', 'quiet': True, 'noplaylist': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return ydl.prepare_filename(info)
    except: return None

def extract_frames(video_path):
    video = cv2.VideoCapture(video_path)
    base64Frames = []
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0: return []
    step = max(1, total_frames // 4) 
    for i in range(0, total_frames, step):
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, frame = video.read()
        if success:
            h, w, _ = frame.shape
            scale = 512 / float(w)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (512, new_h))
            _, buffer = cv2.imencode(".jpg", frame)
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
            if len(base64Frames) >= 4: break
    video.release()
    return base64Frames

def clean_ai_output(text):
    """Membersihkan nomor, tanda kutip, dan label."""
    if not text: return ""
    text = text.strip().strip('"').strip("'")
    text = re.sub(r'^\d+[\.\)]\s*', '', text) # Hapus "1." atau "1)"
    text = re.sub(r'^-\s*', '', text)         # Hapus "- "
    text = text.replace("**Prompt:**", "").replace("Here is the prompt:", "")
    text = text.replace("**", "") # Hapus bold
    return text.strip()

# --- UI HALAMAN ---
st.set_page_config(page_title="Veo 3 Prompter Pro", page_icon="🔥", layout="wide")
st.title("🔥 Veo 3 Prompter (Final)")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Konfigurasi")
    provider = st.radio("Provider:", ["Google Gemini", "OpenAI"])
    api_key = st.text_input(f"API Key {provider}", type="password")
    
    selected_model = ""
    if provider == "Google Gemini":
        gemini_opts = ["gemini-1.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-flash"]
        selected_model = st.selectbox("Model:", gemini_opts)
    elif provider == "OpenAI":
        openai_opts = ["gpt-4o-mini", "gpt-4o", "gpt-5-mini", "gpt-5-nano", "gpt-4.1-mini"]
        selected_model = st.selectbox("Model:", openai_opts)
        
    st.divider()
    
    # REVISI: Menggunakan Number Input (Max 15)
    num_variations = st.number_input(
        "Jumlah Variasi Prompt:", 
        min_value=1, 
        max_value=15, 
        value=1,
        step=1
    )
    st.caption("Maksimal 15 variasi sekaligus.")

# --- INPUT ---
tab1, tab2 = st.tabs(["📂 Upload", "🔗 Link"])
video_path = None
do_process = False

with tab1:
    uf = st.file_uploader("Upload Video", type=["mp4", "mov"])
    if uf:
        t = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        t.write(uf.read())
        video_path = t.name
        st.video(video_path)
        if st.button("Generate (Upload)", type="primary"): do_process = True

with tab2:
    url = st.text_input("Link Video")
    if url and st.button("Generate (Link)", type="primary"):
        with st.spinner("Downloading..."):
            video_path = download_video_from_url(url)
            if video_path: 
                st.video(video_path)
                do_process = True

# --- LOGIKA ---
if do_process and video_path:
    if not api_key:
        st.error("API Key kosong!")
    else:
        st.divider()
        status = st.status("Sedang memproses...", expanded=True)
        
        # MULAI BLOK TRY
        # MULAI BLOK TRY
        try:
            # --- SYSTEM PROMPT (CREATIVE VARIATION MODE) ---
            
            # Instruksi khusus untuk variasi
            variation_instruction = ""
            if num_variations > 1:
                variation_instruction = f"""
                You must generate {num_variations} DISTINCT variations.
                For each variation, KEEP the visual style/characters/mood but **INVENT A NEW SCENE or ACTION**.
                (e.g., If the video shows a cat walking, make prompt 1 about the cat eating, prompt 2 about the cat sleeping).
                Separate variations with '|||'.
                """
            else:
                variation_instruction = "Generate 1 creative evolution of this scene (same style, slightly different action)."
            
            system_msg = f"""
            You are a Creative Video Prompt Director for High-End AI (Veo/Sora).
            
            STEP 1: Analyze the input video to extract its **Core Style, Characters, Lighting, and Mood**.
            STEP 2: Write detailed prompts that maintain that exact style but feature **DIFFERENT ACTIONS or SETTINGS**.
            
            OBJECTIVE:
            The user wants to create a series of videos that look like they belong in the same "collection" or "universe" as the input video, but are NOT identical copies.
            
            CONSTRAINTS:
            1. **Consistency:** Keep the aesthetic (e.g. cinematic, cartoon, lens type) EXACTLY like the reference.
            2. **Diversity:** CHANGE the specific action, angle, or background environment for each prompt.
            3. **Length:** Approx 150-200 words per prompt (Single Paragraph).
            4. **Detail:** Focus heavily on textures, lighting, and camera movement.
            
            STRICT OUTPUT RULES:
            1. {variation_instruction}
            2. Generate ONLY the raw prompt text.
            3. NO numbering (1., 2.), NO bullet points, NO labels.
            4. Start directly with the visual description.
            """
            
            raw_result = ""

            # Request AI (Bagian ini TETAP SAMA seperti sebelumnya)
            if provider == "Google Gemini":
                genai.configure(api_key=api_key)
                v_file = genai.upload_file(video_path)
                while v_file.state.name == "PROCESSING": time.sleep(1); v_file = genai.get_file(v_file.name)
                model = genai.GenerativeModel(selected_model)
                res = model.generate_content([v_file, system_msg])
                raw_result = res.text

            elif provider == "OpenAI":
                client = OpenAI(api_key=api_key)
                frames = extract_frames(video_path)
                msg = [{"role": "user", "content": [{"type": "text", "text": system_msg}, *map(lambda x: {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{x}"}}, frames)]}]
                res = client.chat.completions.create(model=selected_model, messages=msg)
                raw_result = res.choices[0].message.content

            status.update(label="Membersihkan teks...", state="running")

            # --- PEMBERSIHAN (Cleaning) ---
            final_output = ""
            if num_variations == 1:
                final_output = clean_ai_output(raw_result)
            else:
                # Split berdasarkan separator unik kita
                if "|||" in raw_result: parts = raw_result.split("|||")
                else: parts = raw_result.split("\n\n")
                
                cleaned_parts = [clean_ai_output(p) for p in parts if p.strip()]
                final_output = "\n\n".join(cleaned_parts)

            status.update(label="Selesai!", state="complete", expanded=False)

            # --- TAMPILAN HASIL ---
            st.success("✅ Prompt Kreatif Siap:")
            st.code(final_output, language="text", line_numbers=False)
            st.download_button("📥 Download .txt", final_output, "prompt_veo_creative.txt")

        except Exception as e:
            st.error(f"Error: {e}")
        
        finally:
            if video_path and os.path.exists(video_path):
                try: os.unlink(video_path)
                except: pass
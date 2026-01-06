import streamlit as st
import google.generativeai as genai
from openai import OpenAI
import tempfile
import time
import os
import cv2
import base64
import yt_dlp
import re  # Library untuk pembersihan teks (Regex)

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
    """
    Fungsi 'Satpam' untuk membersihkan sampah dari output AI.
    Menghapus nomor (1.), tanda kutip, dan label.
    """
    if not text: return ""
    
    # 1. Hapus tanda kutip di awal/akhir string
    text = text.strip().strip('"').strip("'")
    
    # 2. Hapus pola nomor di awal kalimat (misal "1. ", "1)", "- ")
    # Regex: Mencari angka diikuti titik/kurung di awal baris
    text = re.sub(r'^\d+[\.\)]\s*', '', text)
    text = re.sub(r'^-\s*', '', text)
    
    # 3. Hapus label umum jika AI masih bandel
    text = text.replace("**Prompt:**", "").replace("Prompt:", "").replace("Here is the prompt:", "")
    
    # 4. Hapus markdown bold berlebihan
    text = text.replace("**", "")
    
    return text.strip()

# --- UI HALAMAN ---
st.set_page_config(page_title="Veo 3 Prompter (Clean)", page_icon="✨", layout="wide")
st.title("✨ Veo 3 Prompter (Clean Output)")

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
        openai_opts = ["gpt-4o-mini", "gpt-4o", "gpt-5-mini", "gpt-5-nano", "gpt-4.1-mini", "gpt-4.1-nano"]
        selected_model = st.selectbox("Model:", openai_opts)
        
    # --- REVISI: INPUT ANGKA (MAX 15) ---
    st.divider()
    num_variations = st.number_input(
        "Jumlah Variasi Prompt:", 
        min_value=1, 
        max_value=15, # Batas maksimal jadi 15
        value=1,
        step=1
    )
    st.caption("Tips: Jika memilih >5, proses mungkin agak lama.")

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
        if st.button("Generate Prompt (Upload)", type="primary"): do_process = True

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
        
        try:
            # --- SYSTEM PROMPT (STRICT) ---
            # --- BAGIAN REVISI: SYSTEM PROMPT (VERSION: LONG & DETAILED) ---
            
            # --- BAGIAN REVISI: SYSTEM PROMPT (200 Words Limit) ---
            
            instruction_text = "Generate ONLY the raw prompt text."
            if num_variations > 1:
                instruction_text = f"Generate {num_variations} variations. Separate them with '|||'."
            
            system_msg = f"""
            You are a Professional Video Prompt Engineer.
            Analyze the video frames. Write a detailed visual prompt for Veo/Sora.
            
            CONSTRAINTS:
            1. **Length:** Limit each prompt to approximately **200 words**.
            2. **Format:** Must be a **Single Continuous Paragraph**. Do NOT break into lines.
            3. **Content:** Describe Textures, Lighting, Camera, and Atmosphere vividly.
            
            STRICT OUTPUT RULES:
            1. {instruction_text}
            2. NO numbering, NO bullet points, NO labels like 'Subject:'.
            3. Start directly with the visual description (e.g., 'Cinematic shot of...').
            """
            
            raw_result = ""

            # Request ke AI
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

            # --- TAHAP PEMBERSIHAN (PYTHON CLEANING) ---
            final_output = ""
            
            if num_variations == 1:
                # Bersihkan total
                final_output = clean_ai_output(raw_result)
            else:
                # Jika variasi banyak, pisahkan dulu, bersihkan satu-satu, lalu gabung
                # AI mungkin pakai ||| atau baris baru, kita coba split pintar
                if "|||" in raw_result:
                    parts = raw_result.split("|||")
                else:
                    parts = raw_result.split("\n\n") # Fallback jika AI lupa separator
                
                cleaned_parts = [clean_ai_output(p) for p in parts if p.strip()]
                final_output = "\n\n".join(cleaned_parts)

            status.update(label="Selesai!", state="complete", expanded=False)

            # --- REVISI: HASIL DENGAN TOMBOL COPY ---
            st.success("✅ Prompt Siap:")
            
            # Gunakan st.code() -> Otomatis ada tombol 'COPY' (ikon kertas) di pojok kanan atas kotak
            st.code(final_output, language="text", line_numbers=False)
            
            # Tombol Download tetap ada sebagai cadangan
            st.download_button("📥 Download file .txt", final_output, "prompt_video.txt")
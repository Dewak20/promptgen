import streamlit as st
import google.generativeai as genai
from openai import OpenAI
import tempfile
import time
import os
import cv2
import base64
import yt_dlp
import re

# --- FUNGSI BANTUAN ---
def download_video_from_url(url):
    try:
        # Nama file unik berdasarkan timestamp agar tidak bentrok saat batch
        unique_name = f"temp_{int(time.time())}_{str(url)[-5:]}"
        ydl_opts = {
            'format': 'best[ext=mp4]/best', 
            'outtmpl': f'{unique_name}_%(id)s.%(ext)s', 
            'quiet': True, 
            'noplaylist': True
        }
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
    if not text: return ""
    text = text.strip().strip('"').strip("'")
    text = re.sub(r'^\d+[\.\)]\s*', '', text) 
    text = re.sub(r'^-\s*', '', text)
    text = text.replace("**Prompt:**", "").replace("Here is the prompt:", "").replace("**", "")
    return text.strip()

# --- FUNGSI PROSES INTI (Dipanggil dalam Loop) ---
def process_single_video(video_path, provider, api_key, model_name, num_variations):
    # SYSTEM PROMPT (CREATIVE DIRECTOR)
    variation_instr = ""
    if num_variations > 1:
        variation_instr = f"""
        Generate {num_variations} DISTINCT variations.
        For each variation, KEEP the visual style/characters/mood but **INVENT A NEW SCENE or ACTION**.
        Separate variations with '|||'.
        """
    else:
        variation_instr = "Generate 1 creative evolution of this scene (same style, slightly different action)."
    
    system_msg = f"""
    You are a Creative Video Prompt Director for High-End AI (Veo/Sora).
    
    STEP 1: Analyze the input video Style, Lighting, and Mood.
    STEP 2: Write prompts that maintain that exact style but feature **DIFFERENT ACTIONS/SETTINGS**.
    
    CONSTRAINTS:
    1. **Consistency:** Keep aesthetic EXACTLY like reference.
    2. **Diversity:** CHANGE action/angle for each prompt.
    3. **Length:** Approx 150-200 words (Single Paragraph).
    4. **Detail:** Focus on textures, lighting, camera.
    
    STRICT OUTPUT RULES:
    1. {variation_instr}
    2. Generate ONLY the raw prompt text.
    3. NO numbering, NO bullet points.
    """
    
    raw_result = ""
    
    try:
        if provider == "Google Gemini":
            genai.configure(api_key=api_key)
            v_file = genai.upload_file(video_path)
            # Tunggu processing
            while v_file.state.name == "PROCESSING": 
                time.sleep(1); v_file = genai.get_file(v_file.name)
            
            model = genai.GenerativeModel(model_name)
            res = model.generate_content([v_file, system_msg])
            raw_result = res.text

        elif provider == "OpenAI":
            client = OpenAI(api_key=api_key)
            frames = extract_frames(video_path)
            msg = [{"role": "user", "content": [{"type": "text", "text": system_msg}, *map(lambda x: {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{x}"}}, frames)]}]
            res = client.chat.completions.create(model=model_name, messages=msg)
            raw_result = res.choices[0].message.content
            
        # PEMBERSIHAN
        if num_variations == 1:
            return clean_ai_output(raw_result)
        else:
            if "|||" in raw_result: parts = raw_result.split("|||")
            else: parts = raw_result.split("\n\n")
            cleaned = [clean_ai_output(p) for p in parts if p.strip()]
            return "\n\n".join(cleaned)

    except Exception as e:
        return f"Error: {str(e)}"

# --- UI HALAMAN ---
st.set_page_config(page_title="Veo 3 Batch Prompter", page_icon="📦", layout="wide")
st.title("📦 Veo 3 Batch Prompter")

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
        openai_opts = ["gpt-4o-mini", "gpt-4o", "gpt-5-mini", "gpt-4.1-mini"]
        selected_model = st.selectbox("Model:", openai_opts)
        
    st.divider()
    num_variations = st.number_input("Variasi per Video:", 1, 15, 1)

# --- INPUT (BATCH) ---
tab1, tab2 = st.tabs(["📂 Batch Upload Files", "🔗 Batch Links"])
queue_videos = [] # List antrian: [{'type': 'file'/'path', 'data': ...}]
start_process = False

with tab1:
    # REVISI: accept_multiple_files=True
    uploaded_files = st.file_uploader("Upload Banyak Video Sekaligus", type=["mp4", "mov"], accept_multiple_files=True)
    if uploaded_files and st.button("🚀 Proses Semua File", type="primary"):
        for uf in uploaded_files:
            queue_videos.append({'type': 'file', 'data': uf, 'name': uf.name})
        start_process = True

with tab2:
    # REVISI: Text Area untuk banyak link
    links_text = st.text_area("Paste Daftar Link (Satu link per baris):", height=150, placeholder="https://youtube.com/...\nhttps://tiktok.com/...")
    if links_text and st.button("🚀 Proses Semua Link", type="primary"):
        link_list = [l.strip() for l in links_text.split('\n') if l.strip()]
        for l in link_list:
            queue_videos.append({'type': 'url', 'data': l, 'name': l})
        start_process = True

# --- LOGIKA BATCH PROCESSING ---
if start_process and queue_videos:
    if not api_key:
        st.error("API Key kosong!")
    else:
        st.divider()
        
        # Container untuk menampung hasil
        results_container = st.container()
        combined_text_all = ""
        
        # Progress Bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_items = len(queue_videos)
        
        for index, item in enumerate(queue_videos):
            current_video_path = None
            
            # Update Status
            status_text.write(f"⏳ Memproses video {index+1} dari {total_items}: **{item['name']}**")
            
            try:
                # 1. SIAPKAN FILE DI DISK
                if item['type'] == 'file':
                    t = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    t.write(item['data'].read())
                    current_video_path = t.name
                    t.close()
                elif item['type'] == 'url':
                    current_video_path = download_video_from_url(item['data'])
                
                # 2. PROSES AI
                if current_video_path:
                    prompt_result = process_single_video(
                        current_video_path, provider, api_key, selected_model, num_variations
                    )
                    
                    # 3. TAMPILKAN HASIL PER ITEM
                    with results_container:
                        with st.expander(f"✅ Hasil: {item['name']}", expanded=True):
                            st.code(prompt_result, language="text")
                    
                    # Simpan ke memori gabungan
                    combined_text_all += f"--- SOURCE: {item['name']} ---\n{prompt_result}\n\n{'='*30}\n\n"
                else:
                    st.error(f"Gagal memproses file: {item['name']}")

            except Exception as e:
                st.error(f"Error pada {item['name']}: {e}")
            
            finally:
                # Bersihkan file temp per item
                if current_video_path and os.path.exists(current_video_path):
                    try: os.unlink(current_video_path)
                    except: pass
            
            # Update progress bar
            progress_bar.progress((index + 1) / total_items)

        # SELESAI
        status_text.success("🎉 Semua video selesai diproses!")
        
        # TOMBOL DOWNLOAD ALL
        st.divider()
        st.subheader("📥 Download Gabungan")
        st.download_button(
            label="Download Semua Prompt (.txt)",
            data=combined_text_all,
            file_name=f"batch_prompts_{int(time.time())}.txt",
            mime="text/plain",
            type="primary"
        )
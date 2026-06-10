import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b4
from einops import rearrange
import torchvision.transforms.functional as F
import librosa
import torchvision.models as models
from moviepy import VideoFileClip

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepScan — Deepfake Detector",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL STYLE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #08090d !important;
    color: #e8e6e1 !important;
    font-family: 'Syne', sans-serif !important;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { display: none; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 1.5rem 4rem !important; max-width: 780px !important; }

.hero { text-align: center; padding: 3rem 0 2rem; }
.hero-badge {
    display: inline-block; font-family: 'Space Mono', monospace;
    font-size: 0.65rem; letter-spacing: 0.2em; text-transform: uppercase;
    color: #5bffc8; border: 1px solid #5bffc822; background: #5bffc808;
    padding: 0.3rem 0.9rem; border-radius: 2px; margin-bottom: 1.2rem;
}
.hero-title {
    font-size: clamp(2.4rem, 6vw, 3.8rem); font-weight: 800;
    line-height: 1.05; letter-spacing: -0.03em; color: #ffffff; margin-bottom: 0.6rem;
}
.hero-title span { color: #5bffc8; }
.hero-sub { font-size: 0.95rem; color: #6b7280; font-family: 'Space Mono', monospace; }

.divider { border: none; border-top: 1px solid #1e2028; margin: 1.8rem 0; }

[data-testid="stFileUploader"] {
    border: 1px dashed #2a2d3a !important; border-radius: 8px !important;
    background: #0d0f17 !important; padding: 1rem !important;
}
[data-testid="stFileUploader"]:hover { border-color: #5bffc844 !important; }
[data-testid="stFileUploader"] label {
    color: #9ca3af !important; font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
}

video {
    width: 100% !important; border-radius: 8px !important;
    border: 1px solid #1e2028 !important; margin: 0.5rem 0 !important;
}

.stButton > button {
    width: 100% !important; background: #5bffc8 !important; color: #08090d !important;
    font-family: 'Space Mono', monospace !important; font-size: 0.85rem !important;
    font-weight: 700 !important; letter-spacing: 0.1em !important;
    text-transform: uppercase !important; border: none !important;
    border-radius: 4px !important; padding: 0.85rem 2rem !important;
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }
.stButton > button:active { transform: translateY(0) !important; }

[data-testid="stSpinner"] > div { border-top-color: #5bffc8 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
SEQUENCE_LENGTH = 16        # training'deki sequence_length ile aynı olmalı
IMAGE_SIZE      = 224
HIDDEN_DIM      = 256
DROPOUT         = 0.5
MODEL_PATH      = "en_iyi_modelv3.pth"
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL  (training dosyasıyla birebir aynı)
# ─────────────────────────────────────────────────────────────────────────────
class EfficientNetLSTM(nn.Module):
    def __init__(self, sequence_length=SEQUENCE_LENGTH,
                 hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        self.sequence_length = sequence_length
        backbone = efficientnet_b4(weights=None)
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])
        self.cnn_out_dim = 1792
        self.lstm = nn.LSTM(
            input_size=self.cnn_out_dim, hidden_size=hidden_dim,
            num_layers=2, batch_first=True, bidirectional=True, dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 128), nn.GELU(),
            nn.Dropout(dropout * 0.5), nn.Linear(128, 1),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x     = rearrange(x, 'b t c h w -> (b t) c h w')
        feats = self.cnn(x).flatten(1)
        feats = rearrange(feats, '(b t) d -> b t d', b=B, t=T)
        lstm_out, _ = self.lstm(feats)
        return self.classifier(lstm_out[:, -1, :])


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    model = EfficientNetLSTM().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()
        return model, True
    return model, False


# ─────────────────────────────────────────────────────────────────────────────
#  PREPROCESSING  — training DataGenerator ile birebir aynı
# ─────────────────────────────────────────────────────────────────────────────
inference_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def crop_face(frame_bgr, pad: float = 0.2):
    """
    BGR frame'den yüzü tespit edip kırpar.
    Yüz bulunamazsa None döner.
    pad: yüz bounding box'ına eklenecek kenar payı (yüzün %20'si)
    """
    gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) == 0:
        return None  # Yüz yok → frame'i atla / fallback

    # En büyük yüzü al
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # Padding ekle
    pad_x = int(w * pad)
    pad_y = int(h * pad)
    H, W  = frame_bgr.shape[:2]

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(W, x + w + pad_x)
    y2 = min(H, y + h + pad_y)

    return frame_bgr[y1:y2, x1:x2]


def extract_sequence(video_path: str, seq_len: int = SEQUENCE_LENGTH):
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        return None, []

    # ✅ Training ile birebir aynı: tüm videodan eşit aralıklı linspace
    indices = np.linspace(0, total - 1, seq_len, dtype=int)

    frames_tensor, frames_preview = [], []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break

        face = crop_face(frame)
        
        if face is not None:
            frame_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        else:
            # Yüz bulunamazsa orijinal frame'i kullan (fallback)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pil_img   = Image.fromarray(frame_rgb)
        frames_preview.append(pil_img.resize((160, 90)))
        frames_tensor.append(inference_transform(pil_img))

    cap.release()

    if len(frames_tensor) == 0:
        return None, []

    # Bir frame okunamazsa zero-pad (training'deki break davranışıyla tutarlı)
    while len(frames_tensor) < seq_len:
        frames_tensor.append(torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE))
        frames_preview.append(Image.fromarray(np.zeros((90, 160, 3), dtype=np.uint8)))

    tensor = torch.stack(frames_tensor).unsqueeze(0)   # (1, T, C, H, W)
    return tensor, frames_preview


# ─────────────────────────────────────────────────────────────────────────────
#  INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def predict(model, tensor):
    tensor = tensor.to(DEVICE)
    with torch.no_grad():
        logit = model(tensor).squeeze()
        prob  = torch.sigmoid(logit).item()
    is_fake    = prob >= 0.5
    label      = "FAKE" if is_fake else "REAL"
    confidence = prob if is_fake else (1.0 - prob)
    return label, confidence, prob

# ─────────────────────────────────────────────────────────────────────────────
#  AUDIO
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_audio_model(model_path='best_DenseNet121_FineTuned.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Modelin iskeletini kur (Ağırlıkları yüklemek için)
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Linear(model.last_channel, 128),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(128, 1)
    )
    
    # Eğitilmiş ağırlıkları yükle
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval() # Değerlendirme modu!
    return model, device

def process_audio_for_inference(file_path, audio_length=4*22050):
    audio, sr = librosa.load(file_path, sr=None)
    
    # Kırpma veya Padding (Tahmin için hep baştan kırpıyoruz)
    if len(audio) > audio_length:
        audio = audio[:audio_length]
    else:
        audio = np.pad(audio, (0, audio_length - len(audio)), "constant")
        
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
    
    mfccs_norm = (mfccs - mfccs.mean()) / (mfccs.std() + 1e-6)
    delta_norm = (mfcc_delta - mfcc_delta.mean()) / (mfcc_delta.std() + 1e-6)
    delta2_norm = (mfcc_delta2 - mfcc_delta2.mean()) / (mfcc_delta2.std() + 1e-6)
    
    stacked_features = np.stack([mfccs_norm, delta_norm, delta2_norm], axis=0)
    tensor_features = torch.tensor(stacked_features, dtype=torch.float32)
    resized_features = F.resize(tensor_features, size=(224, 224), antialias=True)
    
    # Model batch beklediği için boyutu (1, 3, 224, 224) yapıyoruz
    return resized_features.unsqueeze(0)

def predict_audio(file_path, model, device):
    input_tensor = process_audio_for_inference(file_path).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item() # 0 ile 1 arası olasılık değeri
        
    if prob >= 0.5:
        return "Fake Audio", prob
    else:
        return "Real Audio", prob
        
# ─────────────────────────────────────────────────────────────────────────────
#  UI — HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">DeepFake<span> Detection</span></div>
    <div class="hero-sub">deepfake tespit sistemi</div>
</div>
<hr class="divider">
""", unsafe_allow_html=True)

model, weights_loaded = load_model()
audio_model, device = load_audio_model('best_model_audio.pth')

if not weights_loaded:
    st.markdown(f"""
    <div style="background:#1a120a; border:1px solid #4a2e0a; border-radius:6px;
                padding:0.9rem 1.1rem; font-family:'Space Mono',monospace;
                font-size:0.72rem; color:#d97706; line-height:1.6;">
        ⚠ Model weights not found at <code>{MODEL_PATH}</code> —
        place your <code>.pth</code> file in the same directory as this app.
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="background:#0d0f17; border:1px solid #1e2028; border-left:3px solid #5bffc830;
            border-radius:4px; padding:0.9rem 1.1rem; font-family:'Space Mono',monospace;
            font-size:0.72rem; color:#6b7280; line-height:1.7; margin-bottom:1.2rem;">
    MODEL &nbsp;&nbsp;&nbsp;&nbsp;·&nbsp; FakeAVCeleb v1.2 — görsel işitsel çoklu model analizi<br>
    GİRDİ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;·&nbsp; 16 kare  · 224×224 · ImageNet normalizasyonu
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  UI — UPLOAD
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
#  UI — UPLOAD
# ─────────────────────────────────────────────────────────────────────────────

video_file = st.file_uploader("Video veya ses dosyasını buraya sürükleyin.", type=["mp4", "avi", "mov", "wav", "mp3"])

if video_file:
    file_ext = video_file.name.split(".")[-1].lower()
    is_audio_only = file_ext in ["wav", "mp3"]

    if is_audio_only:
        st.audio(video_file)
    else:
        st.video(video_file)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    if st.button("⬡  ANALİZİ BAŞLAT", use_container_width=True):
        suffix = f".{file_ext}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(video_file.read())
            tmp_path = tmp.name

        t0 = time.time()

        vid_label, vid_conf, vid_raw_prob = None, 0.0, 0.0
        aud_label, aud_conf, aud_raw_prob = None, 0.0, 0.0
        tensor, preview_frames = None, None
        audio_extracted = False

        if is_audio_only:
            with st.spinner("Ses Analiz Ediliyor..."):
                try:
                    a_lbl_raw, a_prob = predict_audio(tmp_path, audio_model, device)
                    aud_label = a_lbl_raw.upper()
                    aud_raw_prob = a_prob
                    aud_conf = a_prob if "FAKE" in aud_label else (1 - a_prob)
                    audio_extracted = True
                except Exception as e:
                    st.error(f"Ses analizi başarısız: {e}")
        else:
            with st.spinner("Video Analiz Ediliyor & Ses Çıkartılıyor..."):
                tensor, preview_frames = extract_sequence(tmp_path)
                if tensor is not None:
                    v_lbl, v_cnf, v_raw = predict(model, tensor)
                    vid_label = f"{v_lbl} VIDEO"
                    vid_conf = v_cnf
                    vid_raw_prob = v_raw

                try:
                    video_clip = VideoFileClip(tmp_path)
                    if video_clip.audio is not None:
                        tmp_audio_path = "temp_audio_extract.wav"
                        video_clip.audio.write_audiofile(tmp_audio_path, logger=None)
                        video_clip.close()
                        a_lbl_raw, a_prob = predict_audio(tmp_audio_path, audio_model, device)
                        aud_label = a_lbl_raw.upper()
                        aud_raw_prob = a_prob
                        aud_conf = a_prob if "FAKE" in aud_label else (1 - a_prob)
                        audio_extracted = True
                        if os.path.exists(tmp_audio_path):
                            os.remove(tmp_audio_path)
                except Exception:
                    pass

        elapsed = time.time() - t0
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

        # ── TASARIM FONKSİYONU ──
        def render_verdict_card(label, confidence, raw_prob, title_text):
            is_fake        = "FAKE" in label
            conf_pct       = confidence * 100
            fake_pct       = raw_prob * 100
            real_pct       = 100 - fake_pct
            verdict_color  = "#ff5b5b" if is_fake else "#5bffc8"
            verdict_bg     = "#1a0505" if is_fake else "#051a10"
            verdict_border = "#4a1a1a" if is_fake else "#1a4a2e"
            verdict_icon   = "✗" if is_fake else "✓"
            html = f"""
            <div style="background:{verdict_bg}; border:1px solid {verdict_border};
                        border-radius:8px; padding:1.2rem 1.4rem;">
                <div style="font-family:'Space Mono',monospace; font-size:0.58rem;
                            letter-spacing:0.2em; text-transform:uppercase;
                            color:{verdict_color}; margin-bottom:0.3rem;">
                    {title_text}
                </div>
                <div style="font-size:1.5rem; font-weight:800; letter-spacing:-0.02em;
                            color:{verdict_color}; line-height:1.2; margin-bottom:0.6rem;">
                    {verdict_icon} {label}
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:0.25rem;">
                    <span style="font-family:'Space Mono',monospace; font-size:0.62rem;
                                color:#6b7280; text-transform:uppercase;">confidence</span>
                    <span style="font-family:'Space Mono',monospace; font-size:0.75rem;
                                font-weight:700; color:{verdict_color};">{conf_pct:.1f}%</span>
                </div>
                <div style="height:3px; background:#1e2028; border-radius:2px;">
                    <div style="height:100%; width:{conf_pct:.2f}%;
                                background:{verdict_color}; border-radius:2px;"></div>
                </div>
            </div>
            """
            return html, fake_pct, real_pct, is_fake

        cell = "background:#0d0f17; border:1px solid #1e2028; border-radius:6px; padding:0.7rem 0.8rem; text-align:center;"
        lbl  = "font-family:'Space Mono',monospace; font-size:0.52rem; color:#4b5563; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:0.25rem;"
        val  = "font-size:0.95rem; font-weight:600; color:#e8e6e1;"

        # ── SONUÇ GÖSTERME ──
        if is_audio_only:
            if audio_extracted:
                a_html, a_fake_pct, a_real_pct, a_is_fake = render_verdict_card(
                    aud_label, aud_conf, aud_raw_prob, "ses tespit sonucu"
                )
                st.markdown(a_html, unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                c1.markdown(f'<div style="{cell}"><div style="{lbl}">Fake Prob</div><div style="{val}">{a_fake_pct:.2f}%</div></div>', unsafe_allow_html=True)
                c2.markdown(f'<div style="{cell}"><div style="{lbl}">Real Prob</div><div style="{val}">{a_real_pct:.2f}%</div></div>', unsafe_allow_html=True)
                c3.markdown(f'<div style="{cell}"><div style="{lbl}">Süre</div><div style="{val}">{elapsed:.2f}s</div></div>', unsafe_allow_html=True)

                if a_is_fake:
                    st.markdown("""
                    <div style="background:#1a120a; border:1px solid #4a2e0a; border-radius:6px;
                                padding:0.7rem 1rem; font-family:'Space Mono',monospace;
                                font-size:0.68rem; color:#d97706; line-height:1.5; margin-top:0.5rem;">
                        ⚠ Bu seste sentetik manipülasyon belirtisi tespit edildi. Olasılıksal tahmindir.
                    </div>
                    """, unsafe_allow_html=True)
        else:
            if tensor is None:
                st.markdown("""
                <div style="background:#1a0505; border:1px solid #4a1a1a; border-radius:6px;
                            padding:0.9rem 1.1rem; font-family:'Space Mono',monospace;
                            font-size:0.75rem; color:#ff5b5b; margin-top:1rem;">
                    ✗ Video dosyası okunamadı. Geçerli bir format olduğundan emin olun.
                </div>
                """, unsafe_allow_html=True)
            else:
                v_html, v_fake_pct, v_real_pct, v_is_fake = render_verdict_card(
                    vid_label, vid_conf, vid_raw_prob, "video tespit sonucu"
                )

                col_v, col_a = st.columns(2)

                with col_v:
                    st.markdown(v_html, unsafe_allow_html=True)
                    st.markdown(f"""
                    <div style="display:flex; gap:6px; margin-top:6px;">
                        <div style="{cell} flex:1"><div style="{lbl}">Fake</div><div style="{val}">{v_fake_pct:.1f}%</div></div>
                        <div style="{cell} flex:1"><div style="{lbl}">Real</div><div style="{val}">{v_real_pct:.1f}%</div></div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_a:
                    if audio_extracted:
                        a_html, a_fake_pct, a_real_pct, a_is_fake = render_verdict_card(
                            aud_label, aud_conf, aud_raw_prob, "ses tespit sonucu"
                        )
                        st.markdown(a_html, unsafe_allow_html=True)
                        st.markdown(f"""
                        <div style="display:flex; gap:6px; margin-top:6px;">
                            <div style="{cell} flex:1"><div style="{lbl}">Fake</div><div style="{val}">{a_fake_pct:.1f}%</div></div>
                            <div style="{cell} flex:1"><div style="{lbl}">Real</div><div style="{val}">{a_real_pct:.1f}%</div></div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style="display:flex; align-items:center; justify-content:center;
                                    font-family:'Space Mono',monospace; font-size:0.68rem; color:#2a2d3a;
                                    border:1px dashed #1e2028; border-radius:8px; padding:2rem 1rem;
                                    height:100%; margin-top:2px;">
                            [ ses kanalı bulunamadı ]
                        </div>
                        """, unsafe_allow_html=True)

              
                if v_is_fake or (audio_extracted and a_is_fake):
                    st.markdown("""
                    <div style="background:#1a120a; border:1px solid #4a2e0a; border-radius:6px;
                                padding:0.7rem 1rem; font-family:'Space Mono',monospace;
                                font-size:0.68rem; color:#d97706; line-height:1.5; margin-top:0.5rem;">
                        ⚠ Sentetik manipülasyon belirtisi tespit edildi. Olasılıksal tahmindir.
                    </div>
                    """, unsafe_allow_html=True)

                if preview_frames:
                    st.markdown("""
                    <div style="font-family:'Space Mono',monospace; font-size:0.65rem;
                                letter-spacing:0.15em; text-transform:uppercase;
                                color:#4b5563; margin:1.5rem 0 0.5rem;">
                        analiz edilen frame dizisi
                    </div>
                    """, unsafe_allow_html=True)
                    cols = st.columns(len(preview_frames))
                    for col, frame in zip(cols, preview_frames):
                        col.image(frame, use_container_width=True)

else:
    st.markdown("""
    <div style="text-align:center; padding:2.5rem 0; color:#2a2d3a;
                font-family:'Space Mono',monospace; font-size:0.75rem; letter-spacing:0.1em;">
        ANALİZ İÇİN VIDEO YÜKLEYİN
    </div>
    """, unsafe_allow_html=True)
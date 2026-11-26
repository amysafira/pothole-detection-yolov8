import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Pothole Detector", page_icon="🚧",
                   layout="centered", initial_sidebar_state="collapsed")

# ---------- App-like mobile styling ----------
st.markdown("""
<style>
#MainMenu, header, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {visibility: hidden; height: 0;}
.block-container {padding-top: 0.5rem; padding-bottom: 2rem; max-width: 480px;}
.app-bar {background: linear-gradient(135deg, #F5B700, #E08E00); color: #1a1a1a;
          border-radius: 0 0 22px 22px; padding: 22px 20px 18px; margin: -0.5rem -1rem 1.2rem;}
.app-bar h1 {font-size: 1.55rem; margin: 0; padding: 0; color: #1a1a1a;}
.app-bar p {margin: 4px 0 0; font-size: 0.92rem; opacity: 0.85;}
.card {background: #ffffff; border: 1px solid #ececec; border-radius: 18px;
       padding: 16px 18px; margin-bottom: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);}
.stat-row {display: flex; gap: 10px; margin-bottom: 12px;}
.stat {flex: 1; background: #fff; border: 1px solid #ececec; border-radius: 16px;
       padding: 12px 10px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.05);}
.stat .num {font-size: 1.5rem; font-weight: 700; color: #1a1a1a;}
.stat .lbl {font-size: 0.75rem; color: #777; margin-top: 2px;}
.badge {display: inline-block; padding: 6px 14px; border-radius: 999px; font-weight: 700; font-size: 0.95rem;}
.HIGH {background: #FDE2E1; color: #B42318;}
.MEDIUM {background: #FEF0C7; color: #B54708;}
.LOW {background: #D1FADF; color: #067647;}
.ok {background: #D1FADF; color: #067647; border-radius: 16px; padding: 16px; text-align: center; font-weight: 600;}
[data-testid="stFileUploader"] section {border-radius: 18px; padding: 22px; border: 2px dashed #F5B700; background: #FFFBEA;}
[data-testid="stImage"] img {border-radius: 16px;}
.hint {color: #777; font-size: 0.85rem; text-align: center; margin-top: 18px;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    # best.pt is the trained YOLOv8s model downloaded from Colab
    return YOLO("best.pt")


def priority(n, largest_pct):
    if n >= 3 or largest_pct > 5:
        return "HIGH", "Repair urgently"
    if n == 2 or largest_pct > 1.5:
        return "MEDIUM", "Schedule repair soon"
    return "LOW", "Monitor this road"


# ---------- App bar ----------
st.markdown("""
<div class="app-bar">
  <h1>🚧 Pothole Detector</h1>
  <p>Snap a road photo. AI finds potholes and rates repair priority.</p>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("📷 Take or upload a road photo", type=["jpg", "jpeg", "png"])

with st.expander("⚙️ Detection settings"):
    confidence = st.slider("Confidence threshold", 0.10, 0.90, 0.25, 0.05,
                           help="Lower finds more potholes but may include false alarms.")

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    with st.spinner("Scanning road for potholes..."):
        result = load_model().predict(image, imgsz=640, conf=confidence, verbose=False)[0]
    annotated = result.plot(line_width=3)[:, :, ::-1]  # BGR -> RGB

    st.image(annotated, width="stretch")

    n = len(result.boxes)
    if n == 0:
        st.markdown('<div class="ok">✅ No potholes detected. Road looks OK.</div>',
                    unsafe_allow_html=True)
    else:
        h, w = result.orig_shape
        areas = [float((b[2] - b[0]) * (b[3] - b[1])) / (w * h) * 100
                 for b in result.boxes.xyxy]
        level, action = priority(n, max(areas))
        st.markdown(f"""
        <div class="stat-row">
          <div class="stat"><div class="num">{n}</div><div class="lbl">Potholes</div></div>
          <div class="stat"><div class="num">{max(areas):.1f}%</div><div class="lbl">Largest size</div></div>
        </div>
        <div class="card">
          <div style="font-size:0.85rem;color:#777;margin-bottom:6px;">Repair priority</div>
          <span class="badge {level}">{level}</span>
          <span style="margin-left:8px;color:#444;">{action}</span>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🖼️ View original photo"):
        st.image(image, width="stretch")
else:
    st.markdown('<div class="hint">Tip: on your phone, tap the box above and choose '
                '<b>Take Photo</b> to scan a road on the spot.</div>', unsafe_allow_html=True)

st.markdown('<div class="hint">YOLOv8s fine-tuned on ~665 road images</div>',
            unsafe_allow_html=True)

import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Pothole Detector Malaysia", page_icon="🚧", layout="centered")


@st.cache_resource
def load_model():
    # best.pt is the trained YOLOv8s model downloaded from Colab
    return YOLO("best.pt")


def priority(n, largest_pct):
    if n >= 3 or largest_pct > 5:
        return "HIGH", "🔴"
    if n == 2 or largest_pct > 1.5:
        return "MEDIUM", "🟠"
    return "LOW", "🟢"


st.title("🚧 Pothole Detector for Malaysian Roads")
st.write(
    "Upload a photo of a road. A YOLOv8 model marks each pothole and suggests "
    "a repair priority to help local councils plan road maintenance."
)

confidence = st.slider("Detection confidence threshold", 0.10, 0.90, 0.25, 0.05,
                       help="Lower = finds more potholes but may include false alarms.")
uploaded = st.file_uploader("Upload a road photo", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    with st.spinner("Detecting potholes..."):
        result = load_model().predict(image, imgsz=640, conf=confidence, verbose=False)[0]
    annotated = result.plot(line_width=3)[:, :, ::-1]  # BGR -> RGB for display

    col1, col2 = st.columns(2)
    col1.image(image, caption="Original", width="stretch")
    col2.image(annotated, caption="Detected potholes", width="stretch")

    n = len(result.boxes)
    if n == 0:
        st.success("No potholes detected. Road looks OK.")
    else:
        h, w = result.orig_shape
        areas = [float((b[2] - b[0]) * (b[3] - b[1])) / (w * h) * 100
                 for b in result.boxes.xyxy]
        level, icon = priority(n, max(areas))
        m1, m2, m3 = st.columns(3)
        m1.metric("Potholes detected", n)
        m2.metric("Largest pothole", f"{max(areas):.1f}% of image")
        m3.metric("Repair priority", f"{icon} {level}")
else:
    st.info("Upload a road photo to start. Try a photo taken on a Malaysian road!")

st.caption("Model: YOLOv8s fine-tuned on a Kaggle pothole dataset (~665 images).")

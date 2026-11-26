import hashlib
import io
import random
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFont, ImageOps
from ultralytics import YOLO

st.set_page_config(page_title="Pothole Detector", page_icon="🚧",
                   layout="centered", initial_sidebar_state="collapsed")

MY_TZ = ZoneInfo("Asia/Kuala_Lumpur")

# Approximate centre points of common Klang Valley areas (used when a photo has no GPS)
AREAS = {
    "Bangsar": (3.1300, 101.6710), "Cheras": (3.0870, 101.7440),
    "Kepong": (3.2160, 101.6360), "Setapak": (3.1990, 101.7180),
    "Wangsa Maju": (3.2050, 101.7370), "Ampang": (3.1490, 101.7600),
    "Bukit Bintang": (3.1460, 101.7110), "Sentul": (3.1850, 101.6910),
    "Petaling Jaya": (3.1070, 101.6070), "Subang Jaya": (3.0490, 101.5850),
    "Shah Alam": (3.0730, 101.5180), "Puchong": (3.0250, 101.6180),
}
SEV_COLOR = {"HIGH": "#D92D20", "MEDIUM": "#F79009", "LOW": "#12B76A"}
SEV_ACTION = {"HIGH": "Repair urgently (within 48 hours)",
              "MEDIUM": "Schedule repair within 2 weeks",
              "LOW": "Monitor during next inspection"}

# ---------- App-like mobile styling ----------
st.markdown("""
<style>
#MainMenu, header, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {visibility: hidden; height: 0;}
.block-container {padding-top: 0.5rem; padding-bottom: 2rem; max-width: 520px;}
.app-bar {background: linear-gradient(135deg, #F5B700, #E08E00); color: #1a1a1a;
          border-radius: 0 0 22px 22px; padding: 22px 20px 18px; margin: -0.5rem -1rem 1rem;}
.app-bar h1 {font-size: 1.55rem; margin: 0; padding: 0; color: #1a1a1a;}
.app-bar p {margin: 4px 0 0; font-size: 0.92rem; opacity: 0.85;}
.card {background: #fff; border: 1px solid #ececec; border-radius: 18px;
       padding: 14px 16px; margin-bottom: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);}
.stat-row {display: flex; gap: 8px; margin: 10px 0;}
.stat {flex: 1; background: #fff; border: 1px solid #ececec; border-radius: 16px;
       padding: 12px 6px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.05);}
.stat .num {font-size: 1.4rem; font-weight: 700; color: #1a1a1a;}
.stat .lbl {font-size: 0.72rem; color: #777; margin-top: 2px;}
.badge {display: inline-block; padding: 4px 12px; border-radius: 999px; font-weight: 700; font-size: 0.85rem;}
.HIGH {background: #FDE2E1; color: #B42318;}
.MEDIUM {background: #FEF0C7; color: #B54708;}
.LOW {background: #D1FADF; color: #067647;}
.ph-row {display: flex; align-items: center; justify-content: space-between;
         padding: 10px 0; border-bottom: 1px solid #f0f0f0;}
.ph-row:last-child {border-bottom: none;}
.ph-num {width: 30px; height: 30px; border-radius: 50%; color: #fff; font-weight: 700;
         display: inline-flex; align-items: center; justify-content: center; margin-right: 10px;}
.ph-info {flex: 1; font-size: 0.88rem; color: #333;}
.ph-info small {color: #888;}
.ok {background: #D1FADF; color: #067647; border-radius: 16px; padding: 16px; text-align: center; font-weight: 600;}
[data-testid="stFileUploader"] section {border-radius: 18px; padding: 20px; border: 2px dashed #F5B700; background: #FFFBEA;}
[data-testid="stImage"] img {border-radius: 16px;}
.stTabs [data-baseweb="tab-list"] {gap: 4px;}
.stTabs [data-baseweb="tab"] {border-radius: 12px 12px 0 0; padding: 8px 14px;}
.hint {color: #777; font-size: 0.85rem; text-align: center; margin-top: 14px;}
</style>
""", unsafe_allow_html=True)


# ---------- Model and helpers ----------
@st.cache_resource
def load_model():
    # best.pt is the trained YOLOv8s model downloaded from Colab
    return YOLO("best.pt")


@st.cache_data(show_spinner=False)
def detect(img_bytes, confidence):
    """Run YOLO once per (image, threshold) and return plain box data."""
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(img_bytes))).convert("RGB")
    result = load_model().predict(img, imgsz=640, conf=confidence, verbose=False)[0]
    w, h = img.size
    boxes = []
    for xyxy, conf in zip(result.boxes.xyxy, result.boxes.conf):
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        area_pct = (x2 - x1) * (y2 - y1) / (w * h) * 100
        boxes.append({"box": (x1, y1, x2, y2), "conf": float(conf),
                      "area_pct": area_pct, "severity": pothole_severity(area_pct, float(conf))})
    boxes.sort(key=lambda b: b["area_pct"], reverse=True)  # #1 = largest
    return boxes


def pothole_severity(area_pct, conf):
    if area_pct > 3:
        return "HIGH"
    if area_pct > 1:
        return "MEDIUM"
    return "LOW"


def overall_priority(boxes):
    if not boxes:
        return "NONE"
    sevs = [b["severity"] for b in boxes]
    if "HIGH" in sevs or len(boxes) >= 4:
        return "HIGH"
    if "MEDIUM" in sevs or len(boxes) >= 2:
        return "MEDIUM"
    return "LOW"


def draw_numbered(img, boxes):
    out = img.copy()
    d = ImageDraw.Draw(out)
    lw = max(3, out.width // 200)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(18, out.width // 28))
    except OSError:
        font = ImageFont.load_default()
    for i, b in enumerate(boxes, 1):
        x1, y1, x2, y2 = b["box"]
        col = SEV_COLOR[b["severity"]]
        d.rectangle([x1, y1, x2, y2], outline=col, width=lw)
        label = f"#{i}"
        tb = d.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = max(0, y1 - th - 10)
        d.rectangle([x1, ty, x1 + tw + 12, ty + th + 10], fill=col)
        d.text((x1 + 6, ty + 3), label, fill="white", font=font)
    return out


def gps_from_exif(img):
    """Return (lat, lon) from photo EXIF GPS, or None."""
    try:
        gps = img.getexif().get_ifd(0x8825)
        if not gps or 2 not in gps or 4 not in gps:
            return None

        def to_deg(v):
            d, m, s = [float(x) for x in v]
            return d + m / 60 + s / 3600

        lat, lon = to_deg(gps[2]), to_deg(gps[4])
        if gps.get(1) == "S":
            lat = -lat
        if gps.get(3) == "W":
            lon = -lon
        return (lat, lon)
    except Exception:
        return None


def pdf_text(s):
    return s.encode("latin-1", "replace").decode("latin-1")


def make_pdf(scan, annotated):
    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()
    pdf.set_fill_color(245, 183, 0)
    pdf.rect(0, 0, 210, 28, "F")
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_xy(12, 8)
    pdf.cell(0, 10, "Pothole Damage Report")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_xy(12, 17)
    pdf.cell(0, 6, "Generated by Pothole Detector for Malaysian Roads (YOLOv8)")

    pdf.set_xy(12, 34)
    pdf.set_font("Helvetica", "", 11)
    rows = [("Report ID", scan["id"]), ("Date & time", scan["time"]),
            ("Location", scan["location"]),
            ("Coordinates", f"{scan['lat']:.5f}, {scan['lon']:.5f}"),
            ("Potholes detected", str(scan["count"])),
            ("Overall priority", scan["priority"]),
            ("Recommended action", SEV_ACTION.get(scan["priority"], "No action needed"))]
    for k, v in rows:
        pdf.set_x(12)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(48, 7, pdf_text(k))
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 7, pdf_text(v), new_x="LMARGIN", new_y="NEXT")

    img = annotated.copy()
    img.thumbnail((1200, 1200))
    max_w, max_h = 186, 110
    ratio = min(max_w / img.width, max_h / img.height)
    pdf.ln(3)
    pdf.image(img, x=12, w=img.width * ratio, h=img.height * ratio)
    pdf.ln(4)

    if scan["boxes"]:
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_x(12)
        pdf.cell(0, 8, "Pothole details", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_fill_color(240, 240, 240)
        for txt, w in [("No.", 18), ("Severity", 35), ("Size (% of image)", 45), ("Confidence", 35)]:
            pdf.cell(w, 8, txt, border=1, fill=True)
        pdf.ln()
        pdf.set_font("Helvetica", "", 10)
        for i, b in enumerate(scan["boxes"], 1):
            for txt, w in [(f"#{i}", 18), (b["severity"], 35),
                           (f"{b['area_pct']:.1f}%", 45), (f"{b['conf']*100:.0f}%", 35)]:
                pdf.cell(w, 8, txt, border=1)
            pdf.ln()
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_x(12)
    pdf.multi_cell(186, 4, "Severity is estimated from the pothole's size in the photo and should be "
                            "confirmed by an on-site inspection.")
    return bytes(pdf.output())


# ---------- Session history ----------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------- App bar ----------
st.markdown("""
<div class="app-bar">
  <h1>🚧 Pothole Detector</h1>
  <p>Snap a road photo. AI finds potholes, rates them and maps them.</p>
</div>
""", unsafe_allow_html=True)

tab_scan, tab_map, tab_hist = st.tabs(["📷 Scan", "🗺️ Map", "📊 History"])

# ================= SCAN TAB =================
with tab_scan:
    uploaded = st.file_uploader("Take or upload a road photo", type=["jpg", "jpeg", "png"])

    with st.expander("⚙️ Detection settings"):
        confidence = st.slider("Confidence threshold", 0.10, 0.90, 0.25, 0.05,
                               help="Lower finds more potholes but may include false alarms.")

    if uploaded is not None:
        img_bytes = uploaded.getvalue()
        raw = Image.open(io.BytesIO(img_bytes))
        gps = gps_from_exif(raw)
        image = ImageOps.exif_transpose(raw).convert("RGB")

        with st.spinner("Scanning road for potholes..."):
            boxes = detect(img_bytes, confidence)
        annotated = draw_numbered(image, boxes)
        st.image(annotated, width="stretch")

        # Location
        if gps:
            st.success(f"📍 Location from photo GPS: {gps[0]:.5f}, {gps[1]:.5f}")
            area = "Photo GPS"
            lat, lon = gps
        else:
            area = st.selectbox("📍 Where was this photo taken?", list(AREAS.keys()),
                                help="No GPS found in the photo, so choose the nearest area.")
            rnd = random.Random(hashlib.md5(img_bytes).hexdigest())
            lat = AREAS[area][0] + rnd.uniform(-0.008, 0.008)
            lon = AREAS[area][1] + rnd.uniform(-0.008, 0.008)

        n = len(boxes)
        prio = overall_priority(boxes)
        if n == 0:
            st.markdown('<div class="ok">✅ No potholes detected. Road looks OK.</div>',
                        unsafe_allow_html=True)
        else:
            largest = boxes[0]["area_pct"]
            avg_conf = sum(b["conf"] for b in boxes) / n * 100
            st.markdown(f"""
            <div class="stat-row">
              <div class="stat"><div class="num">{n}</div><div class="lbl">Potholes</div></div>
              <div class="stat"><div class="num">{largest:.1f}%</div><div class="lbl">Largest size</div></div>
              <div class="stat"><div class="num">{avg_conf:.0f}%</div><div class="lbl">Avg confidence</div></div>
            </div>
            <div class="card">
              <div style="font-size:0.82rem;color:#777;margin-bottom:6px;">Overall repair priority</div>
              <span class="badge {prio}">{prio}</span>
              <span style="margin-left:8px;color:#444;font-size:0.9rem;">{SEV_ACTION[prio]}</span>
            </div>
            """, unsafe_allow_html=True)

            rows_html = "".join(
                f'<div class="ph-row"><span class="ph-num" style="background:{SEV_COLOR[b["severity"]]}">{i}</span>'
                f'<div class="ph-info"><b>Pothole #{i}</b><br><small>Size {b["area_pct"]:.1f}% of image · '
                f'Confidence {b["conf"]*100:.0f}%</small></div>'
                f'<span class="badge {b["severity"]}">{b["severity"]}</span></div>'
                for i, b in enumerate(boxes, 1))
            st.markdown(f'<div class="card"><div style="font-size:0.82rem;color:#777;">'
                        f'Pothole details</div>{rows_html}</div>', unsafe_allow_html=True)

        # Save to history (once per photo)
        scan_key = hashlib.md5(img_bytes).hexdigest()[:8].upper()
        now = datetime.now(MY_TZ)
        scan = {"id": f"PH-{scan_key}", "time": now.strftime("%d %b %Y, %I:%M %p"),
                "location": area if area != "Photo GPS" else f"GPS {lat:.4f}, {lon:.4f}",
                "lat": lat, "lon": lon, "count": n,
                "priority": prio if n else "NONE", "boxes": boxes}
        existing = next((s for s in st.session_state.history if s["id"] == scan["id"]), None)
        if existing:
            scan["time"] = existing["time"]
            existing.update(scan)
        else:
            st.session_state.history.append(scan)

        st.download_button("📄 Download damage report (PDF)", data=make_pdf(scan, annotated),
                           file_name=f"pothole_report_{scan['id']}.pdf", mime="application/pdf",
                           width="stretch")

        with st.expander("🖼️ View original photo"):
            st.image(image, width="stretch")
    else:
        st.markdown('<div class="hint">Tip: on your phone, tap the box above and choose '
                    '<b>Take Photo</b> to scan a road on the spot.</div>', unsafe_allow_html=True)

# ================= MAP TAB =================
with tab_map:
    hist = st.session_state.history
    if not hist:
        st.info("Scan a road photo first. Each scan appears here as a pin, coloured by repair priority.")
        st.map(pd.DataFrame({"lat": [3.139], "lon": [101.6869]}), zoom=10, size=0)
    else:
        color_map = {**SEV_COLOR, "NONE": "#1570EF"}
        df = pd.DataFrame([{"lat": s["lat"], "lon": s["lon"], "color": color_map[s["priority"]],
                            "size": 120 + 60 * s["count"]} for s in hist])
        st.map(df, latitude="lat", longitude="lon", color="color", size="size", zoom=10)
        st.markdown('<div class="hint">🔴 High &nbsp; 🟠 Medium &nbsp; 🟢 Low &nbsp; 🔵 No potholes'
                    ' &nbsp;·&nbsp; bigger pin = more potholes</div>', unsafe_allow_html=True)

# ================= HISTORY TAB =================
with tab_hist:
    hist = st.session_state.history
    if not hist:
        st.info("No scans yet. Your scans from this session will be listed here with summary stats.")
    else:
        total_ph = sum(s["count"] for s in hist)
        n_high = sum(1 for s in hist if s["priority"] == "HIGH")
        st.markdown(f"""
        <div class="stat-row">
          <div class="stat"><div class="num">{len(hist)}</div><div class="lbl">Roads scanned</div></div>
          <div class="stat"><div class="num">{total_ph}</div><div class="lbl">Potholes found</div></div>
          <div class="stat"><div class="num">{n_high}</div><div class="lbl">High priority</div></div>
        </div>
        """, unsafe_allow_html=True)

        counts = pd.DataFrame({"Priority": ["HIGH", "MEDIUM", "LOW", "NONE"],
                               "Scans": [sum(1 for s in hist if s["priority"] == p)
                                         for p in ["HIGH", "MEDIUM", "LOW", "NONE"]]})
        st.markdown("**Scans by repair priority**")
        st.bar_chart(counts, x="Priority", y="Scans", color="#F5B700", height=220)

        cards = "".join(
            f'<div class="ph-row"><div class="ph-info"><b>{s["location"]}</b><br>'
            f'<small>{s["time"]} · {s["count"]} pothole(s) · {s["id"]}</small></div>'
            f'<span class="badge {s["priority"] if s["priority"] != "NONE" else "LOW"}">'
            f'{s["priority"] if s["priority"] != "NONE" else "CLEAR"}</span></div>'
            for s in reversed(hist))
        st.markdown(f'<div class="card">{cards}</div>', unsafe_allow_html=True)

        csv = pd.DataFrame([{"Report ID": s["id"], "Date & time": s["time"], "Location": s["location"],
                             "Latitude": round(s["lat"], 6), "Longitude": round(s["lon"], 6),
                             "Potholes": s["count"], "Priority": s["priority"]} for s in hist])
        st.download_button("⬇️ Download all scans (CSV)", csv.to_csv(index=False).encode(),
                           file_name="pothole_scans.csv", mime="text/csv", width="stretch")
        if st.button("🗑️ Clear history", width="stretch"):
            st.session_state.history = []
            st.rerun()

st.markdown('<div class="hint">YOLOv8s fine-tuned on ~665 road images · '
            'Severity is estimated from pothole size in the photo</div>', unsafe_allow_html=True)

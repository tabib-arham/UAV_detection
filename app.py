import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw
import pandas as pd
import altair as alt

# =========================================================
# STREAMLIT PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="UAV Object Detection System",
    layout="wide",
)

# =========================================================
# SIMPLE LOGIN CREDENTIALS (DEMO)
# =========================================================
USERS = {
    "admin": "admin123",
    "uav": "uav123"
}

# =========================================================
# SESSION STATE INIT
# =========================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user" not in st.session_state:
    st.session_state.user = None

# =========================================================
# LOGIN PAGE
# =========================================================
if not st.session_state.logged_in:
    st.title("🚁 UAV Surveillance Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if USERS.get(username) == password:
            st.session_state.logged_in = True
            st.session_state.user = username
            st.rerun()
        else:
            st.error("❌ Invalid username or password")

    st.stop()

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("🚁 UAV Dashboard")
st.sidebar.write(f"👤 User: **{st.session_state.user}**")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.25,
    step=0.05
)

if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()

# =========================================================
# LOAD YOLO MODEL (CACHED)
# =========================================================
@st.cache_resource
def load_model():
    return YOLO("weights/best.pt")

model = load_model()

# =========================================================
# MAIN PAGE
# =========================================================
st.title("📡 UAV Object Detection")
st.write("YOLO-based real-time object detection with confidence scores")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# =========================================================
# INFERENCE
# =========================================================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    col1, col2 = st.columns(2)

    confidence_records = []

    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("Detection Result")
        with st.spinner("Running YOLO inference..."):
            results = model(img_np, conf=confidence)[0]

            draw = ImageDraw.Draw(image)

            for box in results.boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                conf_score = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]

                text = f"{label} {conf_score:.2f}"

                draw.rectangle(
                    [x1, y1, x2, y2],
                    outline="red",
                    width=3
                )

                draw.text(
                    (x1, max(0, y1 - 15)),
                    text,
                    fill="yellow"
                )

                # save for histogram
                confidence_records.append({
                    "Class": label,
                    "Confidence": conf_score
                })

            st.image(image, use_column_width=True)

    # =====================================================
    # CONFIDENCE HISTOGRAM
    # =====================================================
    if confidence_records:
        st.subheader("📊 Confidence Distribution of Class Predictions")

        df_conf = pd.DataFrame(confidence_records)

        hist = (
            alt.Chart(df_conf)
            .mark_bar(opacity=0.7)
            .encode(
                x=alt.X("Confidence:Q", bin=alt.Bin(maxbins=10), title="Confidence"),
                y=alt.Y("count()", title="Number of Detections"),
                color=alt.Color("Class:N", legend=alt.Legend(title="Class")),
                tooltip=["Class", "count()"]
            )
            .properties(
                width="container",
                height=350
            )
        )

        st.altair_chart(hist, use_container_width=True)
    else:
        st.info("No detections found to plot confidence histogram.")

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption("🚁 UAV Detection System | YOLO | Streamlit Cloud")

import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="UAV Object Detection",
    layout="wide"
)

# ---------------- SIMPLE AUTH ----------------
USERS = {
    "admin": "admin123",
    "uav": "uav123"
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------------- LOGIN PAGE ----------------
if not st.session_state.logged_in:
    st.title("🚁 UAV Surveillance Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if USERS.get(username) == password:
            st.session_state.logged_in = True
            st.session_state.user = username
            st.experimental_rerun()
        else:
            st.error("Invalid username or password")

    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.title("🚁 UAV Dashboard")
st.sidebar.write(f"User: **{st.session_state.user}**")

conf_thres = st.sidebar.slider(
    "Confidence Threshold",
    0.1, 1.0, 0.25, 0.05
)

if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.experimental_rerun()

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("weights/best.pt")

model = load_model()

# ---------------- MAIN PAGE ----------------
st.title("📡 UAV Object Detection")
st.write("YOLO-based detection with confidence scores")

uploaded = st.file_uploader(
    "Upload UAV Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    img_np = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("Detection Result")
        with st.spinner("Running detection..."):
            results = model(img_np, conf=conf_thres)[0]

            draw = ImageDraw.Draw(image)

            for box in results.boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]

                text = f"{label} {conf:.2f}"

                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                draw.text((x1, max(0, y1 - 15)), text, fill="yellow")

            st.image(image, use_column_width=True)

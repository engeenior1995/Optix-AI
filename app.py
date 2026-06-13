import requests
import pandas as pd
import streamlit as st
from PIL import Image


# =========================
# CONFIG
# =========================

BACKEND_URL = "https://engeenior-optixai.hf.space"

GITHUB_URL = "https://github.com/engeenior1995"
LINKEDIN_URL = "https://www.linkedin.com/in/engeenior1995"

ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png"]
MAX_FILE_SIZE_MB = 5
MIN_CLASSES_REQUIRED = 2
MIN_IMAGES_PER_CLASS = 2


st.set_page_config(
    page_title="Optix AI Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================
# CSS
# =========================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
    --bg: #070A12;
    --card: #0F1422;
    --card2: #151B2E;
    --border: rgba(255,255,255,0.08);
    --text: #F4F7FB;
    --muted: #9AA4B2;
    --blue: #4F8CFF;
    --purple: #9B5CFF;
    --green: #25D366;
    --red: #FF4B4B;
    --yellow: #FFC857;
}

html, body, .stApp {
    background: radial-gradient(circle at top left, rgba(79,140,255,0.14), transparent 35%),
                radial-gradient(circle at bottom right, rgba(155,92,255,0.12), transparent 35%),
                var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}

.block-container {
    max-width: 1400px;
    padding-top: 1.5rem;
}

section[data-testid="stSidebar"] {
    background: #0B0F1A !important;
    border-right: 1px solid var(--border);
}

h1, h2, h3, h4, h5, p, div, label {
    font-family: 'Inter', sans-serif !important;
    color: var(--text) !important;
}

.hero {
    background: linear-gradient(135deg, rgba(79,140,255,0.18), rgba(155,92,255,0.12));
    border: 1px solid var(--border);
    border-radius: 30px;
    padding: 42px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}

.hero-title {
    font-size: 52px;
    line-height: 1.05;
    font-weight: 900;
    letter-spacing: -2px;
}

.hero-title span {
    background: linear-gradient(90deg, var(--blue), var(--purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-sub {
    max-width: 720px;
    font-size: 17px;
    color: var(--muted) !important;
    margin-top: 16px;
    line-height: 1.7;
}

.badges {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 26px;
}

.badge {
    background: rgba(255,255,255,0.06);
    border: 1px solid var(--border);
    padding: 8px 14px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 700;
    color: #DDE7FF !important;
}

.step-card {
    background: rgba(15,20,34,0.88);
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 26px;
    margin-bottom: 22px;
    box-shadow: 0 18px 50px rgba(0,0,0,0.18);
}

.step-head {
    display: flex;
    align-items: center;
    gap: 16px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 18px;
    margin-bottom: 22px;
}

.step-num {
    width: 46px;
    height: 46px;
    border-radius: 15px;
    background: linear-gradient(135deg, var(--blue), var(--purple));
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 900;
    font-size: 18px;
}

.step-title {
    font-size: 22px;
    font-weight: 900;
}

.step-desc {
    color: var(--muted) !important;
    font-size: 14px;
    margin-top: 4px;
}

.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 24px;
}

.stat {
    background: rgba(15,20,34,0.9);
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 22px;
}

.stat-icon {
    font-size: 26px;
    margin-bottom: 10px;
}

.stat-value {
    font-size: 30px;
    font-weight: 900;
}

.stat-label {
    font-size: 13px;
    color: var(--muted) !important;
}

.class-box {
    background: rgba(255,255,255,0.045);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 15px 17px;
    margin-bottom: 10px;
}

.class-title {
    font-size: 15px;
    font-weight: 800;
}

.class-count {
    font-size: 12px;
    color: var(--muted) !important;
    margin-top: 3px;
}

.stButton > button {
    height: 46px;
    border-radius: 14px !important;
    border: 0 !important;
    background: linear-gradient(135deg, var(--blue), var(--purple)) !important;
    color: white !important;
    font-weight: 900 !important;
}

.stButton > button:hover {
    filter: brightness(1.1);
    transform: translateY(-1px);
}

.stTextInput input {
    background: var(--card2) !important;
    border: 1px solid var(--border) !important;
    color: white !important;
    border-radius: 12px !important;
}

div[data-testid="stFileUploader"] {
    background: var(--card2) !important;
    border: 1px dashed rgba(79,140,255,0.38) !important;
    border-radius: 18px !important;
    padding: 12px !important;
}

.stRadio > div {
    background: var(--card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 12px !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: var(--card2) !important;
    border-radius: 16px !important;
    padding: 6px !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--blue), var(--purple)) !important;
    color: white !important;
    border-radius: 12px !important;
}

@media(max-width: 900px) {
    .stat-grid {
        grid-template-columns: repeat(2, 1fr);
    }

    .hero-title {
        font-size: 38px;
    }
}
</style>
""", unsafe_allow_html=True)


# =========================
# SESSION STATE
# =========================

if "model_trained" not in st.session_state:
    st.session_state.model_trained = False

if "train_result" not in st.session_state:
    st.session_state.train_result = None


# =========================
# HELPERS
# =========================

def safe_json_response(response):
    try:
        return response.json()
    except Exception:
        return {"detail": response.text or "Unknown backend error."}


def check_backend():
    try:
        response = requests.get(f"{BACKEND_URL}/classes", timeout=3)
        return response.status_code == 200
    except Exception:
        return False


def get_classes():
    try:
        response = requests.get(f"{BACKEND_URL}/classes", timeout=5)
        if response.status_code == 200:
            return response.json().get("classes", [])
        return []
    except Exception:
        return []


def validate_class_name(class_name):
    if not class_name or not class_name.strip():
        return False, "Please enter a class name."

    class_name = class_name.strip()

    if len(class_name) < 2:
        return False, "Class name must be at least 2 characters."

    if len(class_name) > 30:
        return False, "Class name must be less than 30 characters."

    cleaned = class_name.replace(" ", "").replace("_", "").replace("-", "")

    if not cleaned.isalnum():
        return False, "Use only letters, numbers, spaces, hyphen or underscore."

    return True, class_name


def validate_files(files):
    if not files:
        return False, "Please upload at least one image."

    for file in files:
        ext = file.name.split(".")[-1].lower()

        if ext not in ALLOWED_EXTENSIONS:
            return False, f"{file.name} is not allowed. Use JPG, JPEG or PNG."

        size_mb = len(file.getvalue()) / (1024 * 1024)

        if size_mb > MAX_FILE_SIZE_MB:
            return False, f"{file.name} is larger than {MAX_FILE_SIZE_MB}MB."

        try:
            Image.open(file).verify()
            file.seek(0)
        except Exception:
            return False, f"{file.name} is not a valid image."

    return True, "Images are valid."


def can_train(classes):
    if len(classes) < MIN_CLASSES_REQUIRED:
        return False, "Create at least 2 classes before training."

    for item in classes:
        if item.get("image_count", 0) < MIN_IMAGES_PER_CLASS:
            return False, f"Class '{item.get('class_name')}' needs at least {MIN_IMAGES_PER_CLASS} images."

    return True, "Ready to train."


def upload_files(class_name, uploaded_files):
    files = []

    for uploaded_file in uploaded_files:
        files.append(
            (
                "files",
                (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type
                )
            )
        )

    return requests.post(
        f"{BACKEND_URL}/upload-sample",
        data={"class_name": class_name},
        files=files,
        timeout=90
    )


def upload_camera_image(class_name, camera_image):
    files = [
        (
            "files",
            (
                "camera_sample.jpg",
                camera_image.getvalue(),
                "image/jpeg"
            )
        )
    ]

    return requests.post(
        f"{BACKEND_URL}/upload-sample",
        data={"class_name": class_name},
        files=files,
        timeout=30
    )


def predict_image(image_file):
    files = {
        "file": (
            image_file.name if hasattr(image_file, "name") else "prediction.jpg",
            image_file.getvalue(),
            "image/jpeg"
        )
    }

    return requests.post(
        f"{BACKEND_URL}/predict",
        files=files,
        timeout=60
    )


def download_from_backend(endpoint, filename, label):
    try:
        response = requests.get(f"{BACKEND_URL}/{endpoint}", timeout=30)

        if response.status_code == 200:
            st.download_button(
                label=label,
                data=response.content,
                file_name=filename,
                mime="application/octet-stream",
                use_container_width=True
            )
        else:
            error = safe_json_response(response)
            st.warning(error.get("detail", "File is not available."))

    except Exception:
        st.error("Backend is offline.")


def step_header(num, title, desc):
    st.markdown(
        f"""
        <div class="step-card">
            <div class="step-head">
                <div class="step-num">{num}</div>
                <div>
                    <div class="step-title">{title}</div>
                    <div class="step-desc">{desc}</div>
                </div>
            </div>
        """,
        unsafe_allow_html=True
    )


def close_card():
    st.markdown("</div>", unsafe_allow_html=True)


def show_prediction_result(result):
    predicted = result.get("predicted_class", "Unknown")
    probabilities = result.get("probabilities", {})

    st.success(f"🎯 Prediction Result: {predicted}")

    if probabilities:
        df = pd.DataFrame({
            "Class": list(probabilities.keys()),
            "Probability": list(probabilities.values())
        })

        st.bar_chart(df.set_index("Class"))

        st.markdown("#### Confidence Breakdown")

        for label, probability in probabilities.items():
            col_a, col_b = st.columns([4, 1])

            with col_a:
                st.progress(float(probability) / 100)

            with col_b:
                st.markdown(f"**{probability}%**")


# =========================
# DATA
# =========================

backend_online = check_backend()
classes = get_classes() if backend_online else []

total_classes = len(classes)
total_samples = sum(item.get("image_count", 0) for item in classes)

train_allowed, train_message = can_train(classes)

if total_classes == 0:
    progress_value = 0
elif total_classes < MIN_CLASSES_REQUIRED:
    progress_value = 33
elif not train_allowed:
    progress_value = 66
elif st.session_state.model_trained:
    progress_value = 100
else:
    progress_value = 80


# =========================
# SIDEBAR
# =========================

with st.sidebar:
    st.markdown("""
    <div style="padding:18px 4px 8px;">
        <div style="font-size:34px;">⚡</div>
        <h2 style="margin-bottom:0;">AI Studio</h2>
        <p style="color:#9AA4B2 !important;font-size:13px;">
            OPTIX AI Studio is a beginner-friendly platform that empowers anyone to create custom image classification models without coding. Build your dataset, train your AI, and test predictions all in one place.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if backend_online:
        st.success("Backend Online")
    else:
        st.error("Backend Offline")

    if st.session_state.model_trained:
        st.success("Model Trained")
    else:
        st.warning("Model Not Trained")

    st.markdown("### Progress")
    st.progress(progress_value)

    st.markdown("### Simple Guide")
    st.markdown("""
    1. Add class name  
    2. Upload images / Capture image  
    3. Add another class  
    4. Train model  
    5. Test prediction  
    6. Download model  
    """)

    st.markdown("### Rules")
    st.caption(f"Minimum classes: {MIN_CLASSES_REQUIRED}")
    st.caption(f"Minimum images per class: {MIN_IMAGES_PER_CLASS}")
    st.caption(f"Max image size: {MAX_FILE_SIZE_MB}MB")

    st.markdown("### Links")
    st.markdown(f"[GitHub]({GITHUB_URL})")
    st.markdown(f"[LinkedIn]({LINKEDIN_URL})")


# =========================
# HERO
# =========================

st.markdown("""
<div class="hero">
    <div class="hero-title">
        Build Your Own<br><span>Image Classification AI</span>
    </div>
    <div class="hero-sub">
        A simple, guided AI studio where anyone can create classes, upload images,
        train a model, test predictions, and download the trained model — all from one clean page.
    </div>
    <div class="badges">
        <div class="badge">No ML Expertise Needed</div>
        <div class="badge">Beginner Friendly</div>
        <div class="badge">FastAPI Backend</div>
        <div class="badge">MobileNetV3</div>
        <div class="badge">Streamlit UI</div>
    </div>
</div>
""", unsafe_allow_html=True)


# =========================
# STATS
# =========================

st.markdown(f"""
<div class="stat-grid">
    <div class="stat">
        <div class="stat-icon">📁</div>
        <div class="stat-value">{total_classes}</div>
        <div class="stat-label">Classes Created</div>
    </div>
    <div class="stat">
        <div class="stat-icon">🖼️</div>
        <div class="stat-value">{total_samples}</div>
        <div class="stat-label">Training Images</div>
    </div>
    <div class="stat">
        <div class="stat-icon">🔌</div>
        <div class="stat-value" style="font-size:22px;">{"Online" if backend_online else "Offline"}</div>
        <div class="stat-label">Backend Status</div>
    </div>
    <div class="stat">
        <div class="stat-icon">🧠</div>
        <div class="stat-value" style="font-size:22px;">{"Ready" if st.session_state.model_trained else "Pending"}</div>
        <div class="stat-label">Model Status</div>
    </div>
</div>
""", unsafe_allow_html=True)


main_col, side_col = st.columns([2.2, 1], gap="large")


# =========================
# MAIN WORKFLOW
# =========================

with main_col:
    step_header(
        "1",
        "Create Your Dataset",
        "Start by entering a class name, then upload images or capture one sample from camera."
    )

    class_name = st.text_input(
        "Class Name",
        placeholder="Example: Cat, Dog, Bottle, Thumbs Up"
    )

    source_type = st.radio(
        "Choose Sample Source",
        ["Upload Images", "Capture from Camera"],
        horizontal=True
    )

    if source_type == "Upload Images":
        uploaded_files = st.file_uploader(
            "Upload training images",
            type=ALLOWED_EXTENSIONS,
            accept_multiple_files=True
        )

        if uploaded_files:
            st.info(f"{len(uploaded_files)} image(s) selected.")

        if st.button("Add Images to Class", use_container_width=True):
            valid_name, name_result = validate_class_name(class_name)

            if not backend_online:
                st.error("Backend is not running.")

            elif not valid_name:
                st.error(name_result)

            else:
                valid_images, image_result = validate_files(uploaded_files)

                if not valid_images:
                    st.error(image_result)

                else:
                    try:
                        response = upload_files(name_result, uploaded_files)

                        if response.status_code == 200:
                            st.success(response.json().get("message", "Images uploaded successfully."))
                            st.balloons()
                            st.rerun()
                        else:
                            error = safe_json_response(response)
                            st.error(error.get("detail", "Upload failed."))

                    except requests.exceptions.ConnectionError:
                        st.error("Backend connection failed.")

    else:
        st.info("Enter a class name, capture one image, then click Add Camera Image.")

        camera_sample = st.camera_input(
            "Capture training image",
            key="training_camera_capture"
        )

        if camera_sample:
            st.image(
                Image.open(camera_sample),
                caption="Captured Training Sample",
                width=280
            )

            st.success("1 image captured and ready to add.")

        if st.button("📸 Add Camera Image to Class", use_container_width=True):
            valid_name, name_result = validate_class_name(class_name)

            if not backend_online:
                st.error("Backend is not running.")

            elif not valid_name:
                st.error(name_result)

            elif camera_sample is None:
                st.error("Please capture an image first.")

            else:
                try:
                    response = upload_camera_image(name_result, camera_sample)

                    if response.status_code == 200:
                        st.success("Camera image added successfully.")
                        st.balloons()
                        st.rerun()
                    else:
                        error = safe_json_response(response)
                        st.error(error.get("detail", "Camera upload failed."))

                except requests.exceptions.ConnectionError:
                    st.error("Backend connection failed.")

    close_card()


    step_header(
        "2",
        "Train Your AI Model",
        "After creating at least two classes with images, train your custom model."
    )

    if not backend_online:
        st.error("Backend is offline.")

    elif not train_allowed:
        st.warning(train_message)

    else:
        st.success("Dataset is ready. You can train the model now.")

    if st.button(
        "Train Model Now",
        disabled=(not backend_online or not train_allowed),
        use_container_width=True
    ):
        try:
            with st.spinner("Training model..."):
                response = requests.post(f"{BACKEND_URL}/train", timeout=300)

            if response.status_code == 200:
                result = response.json()
                st.session_state.model_trained = True
                st.session_state.train_result = result
                st.success("Model trained successfully.")
                st.balloons()
            else:
                error = safe_json_response(response)
                st.error(error.get("detail", "Training failed."))

        except requests.exceptions.ConnectionError:
            st.error("Backend connection failed.")

    if st.session_state.train_result:
        result = st.session_state.train_result

        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Accuracy", f"{result.get('accuracy', 0)}%")
        m2.metric("Precision", f"{result.get('precision', 0)}%")
        m3.metric("Recall", f"{result.get('recall', 0)}%")
        m4.metric("F1 Score", f"{result.get('f1_score', 0)}%")

        with st.expander("View training details"):
            st.json(result)

    close_card()


    step_header(
        "3",
        "Test Prediction",
        "Upload or capture a new image and check what your AI model predicts."
    )

    if not st.session_state.model_trained:
        st.warning("Train the model first before prediction.")

    pred_tab, cam_tab = st.tabs(["Upload Test Image", "Capture from Camera"])

    with pred_tab:
        test_image = st.file_uploader(
            "Upload test image",
            type=ALLOWED_EXTENSIONS,
            key="test_image"
        )

        if test_image:
            st.image(Image.open(test_image), caption="Test image", width=320)

        if st.button(
            "Run Prediction",
            disabled=(not backend_online or not st.session_state.model_trained),
            use_container_width=True
        ):
            if test_image is None:
                st.error("Upload a test image first.")
            else:
                try:
                    response = predict_image(test_image)

                    if response.status_code == 200:
                        show_prediction_result(response.json())
                    else:
                        error = safe_json_response(response)
                        st.error(error.get("detail", "Prediction failed."))

                except requests.exceptions.ConnectionError:
                    st.error("Backend connection failed.")

    with cam_tab:
        camera_image = st.camera_input("Take photo for prediction")

        if camera_image:
            st.image(Image.open(camera_image), caption="Captured image", width=320)

        if st.button(
            "Predict Captured Image",
            disabled=(not backend_online or not st.session_state.model_trained),
            use_container_width=True
        ):
            if camera_image is None:
                st.error("Capture an image first.")
            else:
                try:
                    response = predict_image(camera_image)

                    if response.status_code == 200:
                        show_prediction_result(response.json())
                    else:
                        error = safe_json_response(response)
                        st.error(error.get("detail", "Prediction failed."))

                except requests.exceptions.ConnectionError:
                    st.error("Backend connection failed.")

    close_card()


    step_header(
        "4",
        "Download Your Model",
        "Export your trained model and labels for future use."
    )

    if not st.session_state.model_trained:
        st.warning("Train the model first to enable downloads.")
    else:
        d1, d2, d3 = st.columns(3)

        with d1:
            download_from_backend(
                "download-model",
                "teachable_model.pkl",
                "Download Model"
            )

        with d2:
            download_from_backend(
                "download-labels",
                "class_labels.json",
                "Download Labels"
            )

        with d3:
            download_from_backend(
                "download-full-package",
                "teachable_machine_package.zip",
                "Download Full Package"
            )

    close_card()


# =========================
# RIGHT PANEL
# =========================

with side_col:
    step_header(
        "📊",
        "Dataset Panel",
        "Your current classes and sample counts."
    )

    if not backend_online:
        st.error("Backend offline.")

    elif not classes:
        st.info("No classes yet.")

    else:
        for item in classes:
            class_title = item.get("class_name")
            image_count = item.get("image_count", 0)

            st.markdown(
                f"""
                <div class="class-box">
                    <div class="class-title">🏷️ {class_title}</div>
                    <div class="class-count">{image_count} sample(s)</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            with st.expander(f"Manage {class_title}"):
                if st.button(
                    "Delete Class",
                    key=f"delete_{class_title}",
                    use_container_width=True
                ):
                    try:
                        response = requests.delete(
                            f"{BACKEND_URL}/delete-class/{class_title}",
                            timeout=30
                        )

                        if response.status_code == 200:
                            st.session_state.model_trained = False
                            st.session_state.train_result = None
                            st.success("Class deleted.")
                            st.rerun()
                        else:
                            error = safe_json_response(response)
                            st.error(error.get("detail", "Delete failed."))

                    except requests.exceptions.ConnectionError:
                        st.error("Backend connection failed.")

    close_card()


    step_header(
        "⚠️",
        "Reset Project",
        "Delete all classes, images and trained model."
    )

    confirm = st.checkbox("I understand this will delete everything.")

    if st.button(
        "Delete Everything",
        disabled=(not backend_online or not confirm),
        use_container_width=True
    ):
        try:
            response = requests.delete(f"{BACKEND_URL}/delete-all", timeout=60)

            if response.status_code == 200:
                st.session_state.model_trained = False
                st.session_state.train_result = None
                st.success("Everything deleted.")
                st.rerun()
            else:
                error = safe_json_response(response)
                st.error(error.get("detail", "Reset failed."))

        except requests.exceptions.ConnectionError:
            st.error("Backend connection failed.")

    close_card()


# =========================
# FOOTER
# =========================

st.divider()

footer_left, footer_right = st.columns([3, 1])

with footer_left:
    st.markdown("### Muhammad Zarq Ali")
    st.caption("AI Engineer • Machine Learning Developer")

with footer_right:
    st.link_button("🐙 GitHub", GITHUB_URL, use_container_width=True)
    st.link_button("💼 LinkedIn", LINKEDIN_URL, use_container_width=True)

st.caption("All Rights Reserved © 2026.")
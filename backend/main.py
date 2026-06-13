import os
import uuid
import shutil
import logging
import json
import zipfile
import traceback
from typing import List
from io import BytesIO

import joblib
import numpy as np
from PIL import Image, UnidentifiedImageError

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


MIN_SAMPLES_PER_CLASS = 2

# =========================
# LOGGING
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


# =========================
# APP SETUP
# =========================

app = FastAPI(
    title="Optix AI Studio Backend",
    description="Custom image classification backend using MobileNetV3 + Logistic Regression",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error("Unhandled backend error:\n%s", traceback.format_exc())

    return JSONResponse(
        status_code=500,
        content={
            "detail": str(exc),
            "error_type": exc.__class__.__name__,
        },
    )


# =========================
# CONSTANTS
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
LABELS_PATH = os.path.join(MODEL_DIR, "class_labels.json")
PACKAGE_PATH = os.path.join(MODEL_DIR, "optix_ai_model_package.zip")

ALLOWED_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

MAX_FILE_SIZE_MB = 10
MIN_CLASSES = 2
MIN_SAMPLES_PER_CLASS = 2

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# =========================
# FEATURE EXTRACTOR
# =========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = MobileNet_V3_Small_Weights.DEFAULT

feature_extractor = mobilenet_v3_small(weights=weights)
feature_extractor.classifier = torch.nn.Identity()
feature_extractor.to(DEVICE)
feature_extractor.eval()

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

logger.info("MobileNetV3-Small feature extractor loaded on %s.", DEVICE)


# =========================
# HELPERS
# =========================

def validate_class_name(class_name: str) -> str:
    class_name = class_name.strip()

    if not class_name:
        raise HTTPException(status_code=400, detail="Class name cannot be empty.")

    if len(class_name) > 64:
        raise HTTPException(
            status_code=400,
            detail="Class name must be 64 characters or fewer.",
        )

    if any(c in class_name for c in ["/", "\\", "..", "\x00"]):
        raise HTTPException(
            status_code=400,
            detail="Class name contains invalid characters.",
        )

    return class_name


def is_allowed_extension(filename: str) -> bool:
    if not filename or "." not in filename:
        return False

    ext = filename.rsplit(".", 1)[-1].lower()
    return ext in ALLOWED_EXTENSIONS


def read_image(image_bytes: bytes) -> Image.Image:
    try:
        return Image.open(BytesIO(image_bytes)).convert("RGB")

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Cannot decode image. File may be corrupt.",
        )

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Image read error: {str(e)}",
        )


def verify_image(image_bytes: bytes) -> bool:
    try:
        Image.open(BytesIO(image_bytes)).verify()
        return True
    except Exception:
        return False


def extract_features(image: Image.Image) -> np.ndarray:
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        features = feature_extractor(image_tensor)

    return features.cpu().numpy().flatten()


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(
            status_code=400,
            detail="No trained model found. Please train the model first.",
        )

    return joblib.load(MODEL_PATH)


def remove_existing_model():
    for path in [MODEL_PATH, LABELS_PATH, PACKAGE_PATH]:
        if os.path.exists(path):
            os.remove(path)


def get_dataset_summary():
    classes = []

    if not os.path.exists(DATASET_DIR):
        return classes

    for class_name in sorted(os.listdir(DATASET_DIR)):
        class_path = os.path.join(DATASET_DIR, class_name)

        if not os.path.isdir(class_path):
            continue

        image_files = [
            file
            for file in os.listdir(class_path)
            if not file.startswith(".")
            and is_allowed_extension(file)
        ]

        classes.append(
            {
                "class_name": class_name,
                "image_count": len(image_files),
            }
        )

    return classes


def save_labels_metadata(classes, total_images):
    metadata = {
        "app_name": "Optix AI Studio",
        "classes": sorted(classes),
        "total_classes": len(classes),
        "total_images": int(total_images),
        "model_type": "MobileNetV3-Small + Logistic Regression",
        "feature_extractor": "torchvision.models.mobilenet_v3_small",
        "classifier": "sklearn.linear_model.LogisticRegression",
        "image_size": "224x224",
        "framework": "PyTorch + Scikit-Learn",
    }

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


def create_model_package():
    if os.path.exists(PACKAGE_PATH):
        os.remove(PACKAGE_PATH)

    with zipfile.ZipFile(PACKAGE_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
        if os.path.exists(MODEL_PATH):
            zipf.write(MODEL_PATH, arcname="model.pkl")

        if os.path.exists(LABELS_PATH):
            zipf.write(LABELS_PATH, arcname="class_labels.json")

    return PACKAGE_PATH


def make_json_safe(data):
    return json.loads(json.dumps(data, default=float))


# =========================
# ROUTES
# =========================

@app.get("/", tags=["Health"])
def health_check():
    return {
        "status": "ok",
        "message": "Optix AI Studio backend is running.",
        "version": "3.0.0",
        "device": str(DEVICE),
    }


@app.get("/classes", tags=["Dataset"])
def get_classes():
    return {
        "classes": get_dataset_summary(),
    }


@app.post("/upload-sample", tags=["Dataset"])
async def upload_sample(
    class_name: str = Form(...),
    files: List[UploadFile] = File(...),
):
    class_name = validate_class_name(class_name)

    class_folder = os.path.join(DATASET_DIR, class_name)
    os.makedirs(class_folder, exist_ok=True)

    saved_files = []
    skipped = []
    
    for class_name in class_names:
    class_folder = os.path.join(DATASET_DIR, class_name)

    image_files = [
        f for f in os.listdir(class_folder)
        if not f.startswith(".")
    ]

    if len(image_files) < 10:
        raise HTTPException(
            status_code=400,
            detail=f"Class '{class_name}' must contain at least 10 images."
        )
    for file in files:
        if not file.filename:
            skipped.append("Unnamed file")
            continue

        if file.content_type not in ALLOWED_TYPES:
            skipped.append(f"{file.filename} unsupported type: {file.content_type}")
            continue

        if not is_allowed_extension(file.filename):
            skipped.append(f"{file.filename} unsupported extension")
            continue

        image_bytes = await file.read()

        if len(image_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
            skipped.append(f"{file.filename} exceeds {MAX_FILE_SIZE_MB}MB")
            continue

        if not verify_image(image_bytes):
            skipped.append(f"{file.filename} corrupt or unreadable")
            continue

        ext = file.filename.rsplit(".", 1)[-1].lower()
        filename = f"{uuid.uuid4()}.{ext}"
        filepath = os.path.join(class_folder, filename)

        with open(filepath, "wb") as f:
            f.write(image_bytes)

        saved_files.append(filename)

    if not saved_files:
        raise HTTPException(
            status_code=400,
            detail=f"No valid images uploaded. Skipped: {', '.join(skipped)}",
        )

    remove_existing_model()

    logger.info("Uploaded %d image(s) for class '%s'.", len(saved_files), class_name)

    return {
        "message": f"{len(saved_files)} image(s) uploaded successfully for '{class_name}'.",
        "saved_files": saved_files,
        "skipped": skipped,
        "model_reset": True,
    }


@app.post("/train", tags=["Model"])
def train_model():
    class_names = [
        folder
        for folder in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, folder))
    ]

    if len(class_names) < MIN_CLASSES:
        raise HTTPException(
            status_code=400,
            detail=f"At least {MIN_CLASSES} classes are required for training.",
        )
    for class_name in class_names:
        class_folder = os.path.join(DATASET_DIR, class_name)

        image_files = [
            f for f in os.listdir(class_folder)
            if not f.startswith(".")
        ]

        if len(image_files) < 10:
            raise HTTPException(
                status_code=400,
                detail=f"Class '{class_name}' must contain at least 10 images."
            )

    X = []
    y = []
    skipped_files = {}

    for class_name in class_names:
        class_folder = os.path.join(DATASET_DIR, class_name)

        image_files = [
            file
            for file in os.listdir(class_folder)
            if not file.startswith(".")
            and is_allowed_extension(file)
        ]

        if len(image_files) < MIN_SAMPLES_PER_CLASS:
            raise HTTPException(
                status_code=400,
                detail=f"Class '{class_name}' needs at least {MIN_SAMPLES_PER_CLASS} images.",
            )

        loaded_count = 0

        for image_file in image_files:
            image_path = os.path.join(class_folder, image_file)

            try:
                image = Image.open(image_path).convert("RGB")
                features = extract_features(image)

                X.append(features)
                y.append(class_name)
                loaded_count += 1

            except Exception as e:
                skipped_files[image_file] = str(e)

        if loaded_count < MIN_SAMPLES_PER_CLASS:
            raise HTTPException(
                status_code=400,
                detail=f"Class '{class_name}' has fewer than {MIN_SAMPLES_PER_CLASS} readable images.",
            )

    X = np.array(X)
    y = np.array(y)

    unique_classes = sorted(set(y))

    if len(unique_classes) < MIN_CLASSES:
        raise HTTPException(
            status_code=400,
            detail="Training requires at least 2 valid classes.",
        )

    use_test_split = True

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y,
        )

        if len(set(y_train)) < MIN_CLASSES or len(set(y_test)) < MIN_CLASSES:
            use_test_split = False

    except ValueError:
        use_test_split = False

    if not use_test_split:
        logger.warning(
            "Dataset too small for safe train/test split. Using full dataset for evaluation."
        )

        X_train = X
        y_train = y
        X_test = X
        y_test = y

    model = LogisticRegression(
        max_iter=2000,
        C=1.0,
        solver="lbfgs",
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    report = make_json_safe(report)

    joblib.dump(model, MODEL_PATH)
    save_labels_metadata(unique_classes, len(X))
    create_model_package()

    logger.info("Model trained successfully. Classes: %s", unique_classes)

    return {
        "message": "Model trained and saved successfully.",
        "classes": unique_classes,
        "total_images": int(len(X)),
        "train_images": int(len(X_train)),
        "test_images": int(len(X_test)),
        "accuracy": round(float(accuracy) * 100, 2),
        "precision": round(float(precision) * 100, 2),
        "recall": round(float(recall) * 100, 2),
        "f1_score": round(float(f1) * 100, 2),
        "evaluation_note": (
            "Dataset was too small for a safe train/test split, so evaluation used the full dataset."
            if not use_test_split
            else "Evaluation used train/test split."
        ),
        "skipped_files": skipped_files if skipped_files else None,
        "classification_report": report,
    }


@app.post("/predict", tags=["Model"])
async def predict(file: UploadFile = File(...)):
    model = load_model()

    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file has no filename.")

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Use JPG, PNG or WEBP.",
        )

    if not is_allowed_extension(file.filename):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file extension.",
        )

    image_bytes = await file.read()

    if len(image_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds {MAX_FILE_SIZE_MB}MB limit.",
        )

    image = read_image(image_bytes)
    features = extract_features(image).reshape(1, -1)

    predicted_class = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    result = {
        cls: round(float(prob) * 100, 2)
        for cls, prob in zip(model.classes_, probabilities)
    }

    result = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))

    logger.info("Prediction: %s", predicted_class)

    return {
        "predicted_class": predicted_class,
        "confidence": result[predicted_class],
        "probabilities": result,
    }


@app.get("/download-model", tags=["Download"])
def download_model():
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(
            status_code=404,
            detail="Model not found. Train the model first.",
        )

    return FileResponse(
        MODEL_PATH,
        filename="optix_model.pkl",
        media_type="application/octet-stream",
    )


@app.get("/download-labels", tags=["Download"])
def download_labels():
    if not os.path.exists(LABELS_PATH):
        raise HTTPException(
            status_code=404,
            detail="Labels file not found. Train the model first.",
        )

    return FileResponse(
        LABELS_PATH,
        filename="class_labels.json",
        media_type="application/json",
    )


@app.get("/download-full-package", tags=["Download"])
def download_full_package():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        raise HTTPException(
            status_code=404,
            detail="Model package not ready. Train the model first.",
        )

    create_model_package()

    return FileResponse(
        PACKAGE_PATH,
        filename="optix_ai_model_package.zip",
        media_type="application/zip",
    )


@app.delete("/delete-class/{class_name}", tags=["Dataset"])
def delete_class(class_name: str):
    class_name = validate_class_name(class_name)
    class_path = os.path.join(DATASET_DIR, class_name)

    if not os.path.exists(class_path):
        raise HTTPException(
            status_code=404,
            detail=f"Class '{class_name}' not found.",
        )

    shutil.rmtree(class_path)
    remove_existing_model()

    logger.info("Deleted class '%s' and reset model.", class_name)

    return {
        "message": f"Class '{class_name}' deleted successfully. Model reset.",
    }


@app.delete("/delete-all", tags=["Dataset"])
def delete_all():
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)

    os.makedirs(DATASET_DIR, exist_ok=True)

    remove_existing_model()

    logger.info("Full reset completed.")

    return {
        "message": "All classes, samples and trained model deleted successfully.",
    }
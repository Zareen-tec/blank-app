import os
import cv2
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from keras.applications import MobileNetV2
from keras import layers, models
from pyngrok import ngrok
from PIL import Image
import subprocess

# 🚀 SET DATASET PATH (Update this as needed)
dataset_path = "dataset"
train_folder = "pro_dataset/train"
test_folder = "pro_dataset/test"

# Create folders if they don't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# ✅ 1. Load images and assign labels
def load_images_and_labels(dataset_path, limit=500):
    images, filenames, labels = [], [], []
    image_files = []

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith((".jpg", ".png", ".jpeg")):
                image_files.append(os.path.join(root, file))
            if len(image_files) >= limit:
                break
        if len(image_files) >= limit:
            break

    print(f"Found {len(image_files)} images. Loading...")

    for file_path in image_files:
        img = cv2.imread(file_path)
        if img is not None:
            images.append(img)
            filenames.append(os.path.basename(file_path))
            label = 1 if "melanoma" in file_path.lower() else 0
            labels.append(label)

    if len(images) == 0:
        raise ValueError("No images found! Check dataset path.")

    return images, filenames, np.array(labels)

# ✅ 2. Preprocess images
def preprocess_images(images):
    processed_images = []
    for img in images:
        img_resized = cv2.resize(img, (224, 224))
        img_normalized = img_resized / 255.0
        processed_images.append(img_normalized)
    return np.array(processed_images)

# ✅ 3. Split and save dataset
def split_and_save_dataset(X, y, filenames, test_size=0.2):
    X_train, X_test, y_train, y_test, train_filenames, test_filenames = train_test_split(
        X, y, filenames, test_size=test_size, random_state=42
    )

    for img, name in zip(X_train, train_filenames):
        cv2.imwrite(os.path.join(train_folder, name), (img * 255).astype(np.uint8))
    for img, name in zip(X_test, test_filenames):
        cv2.imwrite(os.path.join(test_folder, name), (img * 255).astype(np.uint8))

    return X_train, X_test, y_train, y_test

# ✅ 4. Load and preprocess dataset
images, filenames, labels = load_images_and_labels(dataset_path, limit=500)
processed_images = preprocess_images(images)
X_train, X_test, y_train, y_test = split_and_save_dataset(processed_images, labels, filenames)

# ✅ 5. Build MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ✅ 6. Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=15,
    batch_size=32
)

# ✅ 7. Save the trained model
model_path = "skin_cancer_model.h5"
model.save(model_path)
print(f"✅ Model saved successfully at {model_path}!")

# ✅ 8. Streamlit App Interface
def streamlit_interface():
    st.title("🧪 Skin Cancer Detection (Upload Image)")

    uploaded_image = st.file_uploader("Upload a skin image...", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        img = Image.open(uploaded_image)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_input = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_input)
        result = "Melanoma" if prediction[0][0] > 0.5 else "Non-melanoma"

        st.subheader(f"Prediction: {result}")

# ✅ 9. Run Streamlit App
if __name__ == "__main__":
    streamlit_interface()

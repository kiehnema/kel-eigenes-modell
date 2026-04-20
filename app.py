import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from keras.models import load_model

# ----------------------------
# 🌿 App Setup
# ----------------------------
st.set_page_config(page_title="🌿 Pflanzen KI", page_icon="🌱")
st.title("🌿 Wildpflanzen & Bodenanalyse (Teachable Machine)")

st.write("Lade ein Bild einer Pflanze hoch.")

# ----------------------------
# 🤖 Modell laden
# ----------------------------
@st.cache_resource
def load_tm_model():
    model = load_model("keras_model.h5", compile=False)
    class_names = open("labels.txt", "r").readlines()
    return model, class_names

model, class_names = load_tm_model()

# ----------------------------
# 🌱 Bodenlogik (DEIN SYSTEM)
# ----------------------------
category_to_soil = {
    "stickstoffreich": "sehr nährstoffreich",
    "stickstoffarm": "nährstoffarm",
    "trocken": "sandig / trocken",
    "feucht": "feuchter Boden",
    "schattig": "humusreich / schattig"
}

soil_to_plants = {
    "sehr nährstoffreich": ["Tomate", "Kohl", "Zucchini"],
    "nährstoffarm": ["Erbsen", "Kräuter"],
    "sandig / trocken": ["Lavendel", "Rosmarin"],
    "feuchter Boden": ["Gurke", "Kohl"],
    "humusreich / schattig": ["Farne", "Waldpflanzen"]
}

# ----------------------------
# 📷 Upload
# ----------------------------
uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "png", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True)

    st.write("🔍 Analysiere...")

    # ----------------------------
    # 🧠 Bild vorbereiten
    # ----------------------------
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # ----------------------------
    # 🤖 Prediction
    # ----------------------------
    prediction = model.predict(data)
    index = np.argmax(prediction)

    class_name = class_names[index].strip()
    confidence = prediction[0][index]

    st.subheader("🌿 Ergebnis:")

    st.success(f"{class_name} ({round(confidence*100,2)}%)")

    # ----------------------------
    # 🌱 Boden ableiten
    # ----------------------------
    label = class_name.lower()

    soil = "unbekannt"

    for key in category_to_soil:
        if key in label:
            soil = category_to_soil[key]

    st.subheader("🌱 Bodenanalyse")
    st.info(soil)

    # ----------------------------
    # 🌿 Empfehlungen
    # ----------------------------
    st.subheader("🌿 Pflanzenempfehlungen")

    for plant in soil_to_plants.get(soil, []):
        st.write("🌿", plant)

    # ----------------------------
    # 💡 Erklärung
    # ----------------------------
    st.subheader("💡 Erklärung")
    st.write(
        "Das selbst trainierte Teachable-Machine Modell erkennt eine Umweltkategorie. "
        "Diese wird genutzt, um Bodenbedingungen und passende Pflanzen abzuleiten."
    )

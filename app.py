import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

# ----------------------------
# 🌿 App Setup
# ----------------------------
st.set_page_config(page_title="🌿 Wildpflanzen KI", page_icon="🌱")
st.title("🌿 Wildpflanzen & Bodenanalyse (Teachable Machine)")

st.write("Lade ein Bild einer Pflanze hoch und erhalte eine Bodenanalyse.")

# ----------------------------
# 🤖 Modell laden (STABIL)
# ----------------------------
@st.cache_resource
def load_tm_model():
    model = tf.keras.models.load_model("keras_Model.h5", compile=False)
    class_names = open("labels.txt", "r").readlines()
    return model, class_names

model, class_names = load_tm_model()

# ----------------------------
# 🧠 Normalisierung (Pflanzen vereinheitlichen)
# ----------------------------
def normalize(label):
    label = label.lower()

    if "urtica" in label or "nettle" in label:
        return "nettle"
    if "taraxacum" in label or "dandelion" in label:
        return "dandelion"
    if "trifolium" in label or "clover" in label:
        return "clover"
    if "lamium" in label:
        return "lamium"

    return "unknown"

# ----------------------------
# 🌱 Bodenlogik
# ----------------------------
plant_to_soil = {
    "nettle": "stickstoffreich, feucht",
    "dandelion": "nährstoffreich",
    "clover": "stickstoffarm",
    "lamium": "humusreich, schattig"
}

soil_to_plants = {
    "stickstoffreich, feucht": ["Kohl", "Gurke"],
    "nährstoffreich": ["Tomate", "Zucchini"],
    "stickstoffarm": ["Erbsen", "Lavendel"],
    "humusreich, schattig": ["Farne", "Waldpflanzen"]
}

# ----------------------------
# 📷 Bild Upload
# ----------------------------
uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "png", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Dein Bild", use_container_width=True)

    st.write("🔍 Analysiere Pflanze...")

    # ----------------------------
    # 🧠 Bild vorbereiten (224x224)
    # ----------------------------
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)
    normalized_image = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image

    # ----------------------------
    # 🤖 Prediction
    # ----------------------------
    prediction = model.predict(data)
    index = np.argmax(prediction)

    class_name = class_names[index].strip()
    confidence = prediction[0][index]

    # ----------------------------
    # 🌿 Ergebnis anzeigen
    # ----------------------------
    st.subheader("🌿 KI Ergebnis")
    st.success(f"{class_name} ({round(confidence*100, 2)}%)")

    # ----------------------------
    # 🧠 Normalisieren
    # ----------------------------
    plant = normalize(class_name)

    st.subheader("🌱 Erkannte Pflanzenkategorie")
    st.info(plant)

    # ----------------------------
    # 🌱 Boden
    # ----------------------------
    soil = plant_to_soil.get(plant, "unbekannt")

    st.subheader("🌱 Bodenanalyse")
    st.warning(soil)

    # ----------------------------
    # 🌿 Empfehlungen
    # ----------------------------
    st.subheader("🌿 Pflanzempfehlungen")

    for p in soil_to_plants.get(soil, []):
        st.write("🌿", p)

    # ----------------------------
    # 💡 Erklärung
    # ----------------------------
    st.subheader("💡 Erklärung")
    st.write(
        "Das Teachable-Machine Modell erkennt eine Pflanze. "
        "Diese wird in eine ökologische Kategorie übersetzt, "
        "die dann zur Bodenanalyse und Pflanzenempfehlung genutzt wird."
    )

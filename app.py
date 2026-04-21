import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

# ----------------------------
# 🌿 App Setup
# ----------------------------
st.set_page_config(page_title="🌿 Wildpflanzen KI", page_icon="🌱")
st.title("🌿 Wildpflanzen & Bodenanalyse")
st.write("Lade ein Bild einer Pflanze hoch und erhalte eine Bodenanalyse.")

# ----------------------------
# 🤖 Modell laden (STABIL)
# ----------------------------
@st.cache_resource
def load_tm_model():
    model = tf.keras.models.load_model("keras_model.h5", compile=False)
    class_names = open("labels.txt", "r").readlines()
    return model, class_names

model, class_names = load_tm_model()

# ----------------------------
# 🧠 Normalisierung
# ----------------------------
def normalize(label):
    label = label.lower()

    if "urtica" in label or "nettle" in label:
        return "brennnessel"
    if "taraxacum" in label or "dandelion" in label:
        return "loewenzahn"
    if "trifolium" in label or "clover" in label:
        return "klee"
    if "lamium" in label:
        return "taubnessel"

    return "unbekannt"

# ----------------------------
# 🌱 Bodenlogik
# ----------------------------
plant_to_soil = {
    "brennnessel": "stickstoffreich, feucht",
    "loewenzahn": "nährstoffreich",
    "klee": "stickstoffarm",
    "taubnessel": "humusreich, schattig"
}

soil_to_plants = {
    "stickstoffreich, feucht": ["Kohl", "Gurke"],
    "nährstoffreich": ["Tomate", "Zucchini"],
    "stickstoffarm": ["Erbsen", "Lavendel"],
    "humusreich, schattig": ["Farne", "Waldpflanzen"]
}

# ----------------------------
# 📷 Upload
# ----------------------------
uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "png", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Dein Bild", use_container_width=True)

    st.write("🔍 Analysiere...")

    # ----------------------------
    # 🧠 Bild vorbereiten
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
    # 🌿 Ergebnis
    # ----------------------------
    st.subheader("🌿 KI Ergebnis")
    st.success(f"{class_name} ({round(confidence*100, 2)}%)")

    # ----------------------------
    # 🧠 Kategorie
    # ----------------------------
    plant = normalize(class_name)

    st.subheader("🌱 Pflanzenkategorie")
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
        "Das Modell erkennt eine Pflanze. Diese wird einer Kategorie zugeordnet, "
        "aus der Bodenbedingungen und passende Pflanzen abgeleitet werden."
    )

import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# =============================
# SEITE
# =============================
st.set_page_config(page_title="🌿 Pflanzen KI", layout="centered")

st.title("🌿 Pflanzen & Bodenanalyse")

# =============================
# MODELL LADEN (WIE BEI DIR)
# =============================
@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model("keras_model.h5", compile=False)
    with open("labels.txt", "r") as f:
        class_names = f.readlines()
    return model, class_names

model, class_names = load_model_and_labels()

# =============================
# NORMALISIERUNG
# =============================
def normalize(label):
    label = label.lower()

    if "urtica" in label:
        return "brennnessel"
    if "taraxacum" in label:
        return "loewenzahn"
    if "trifolium" in label:
        return "klee"
    if "lamium" in label:
        return "taubnessel"

    return "unbekannt"

# =============================
# BODENLOGIK
# =============================
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

# =============================
# UPLOAD
# =============================
uploaded_file = st.file_uploader("📷 Pflanze hochladen", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Dein Bild", use_column_width=True)

    # ----------------------------
    # PREPROCESSING (IDENTISCH)
    # ----------------------------
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)
    normalized = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized

    # ----------------------------
    # PREDICTION
    # ----------------------------
    with st.spinner("🔍 Analysiere Pflanze..."):
        prediction = model.predict(data)

    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence = float(prediction[0][index])

    st.success(f"🌿 Erkannt: {class_name}")
    st.write(f"Sicherheit: {round(confidence*100, 2)} %")

    # ----------------------------
    # KATEGORIE
    # ----------------------------
    plant = normalize(class_name)

    st.subheader("🌱 Kategorie")
    st.info(plant)

    # ----------------------------
    # BODEN
    # ----------------------------
    soil = plant_to_soil.get(plant, "unbekannt")

    st.subheader("🌱 Bodenanalyse")
    st.warning(soil)

    # ----------------------------
    # EMPFEHLUNG
    # ----------------------------
    st.subheader("🌿 Pflanzenempfehlungen")

    for p in soil_to_plants.get(soil, []):
        st.write("🌿", p)

import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
from supabase import create_client

# =============================
# CONFIG
# =============================
st.set_page_config(page_title="RootWise", layout="wide")

# =============================
# STYLE
# =============================
st.markdown("""
<style>

/* Hintergrund */
.stApp {
    background-color: #E8F5E9;
    color: black;
}

/* GLOBAL: Text wirklich überall schwarz */
html, body, [class*="css"]  {
    color: black !important;
}

/* Titel */
h1, h2, h3, h4, h5, h6, p, span, div {
    color: black !important;
}

/* Buttons (Rosa) */
.stButton>button {
    background-color: #F8BBD0;
    color: black !important;
    border-radius: 10px;
    padding: 10px 16px;
    font-size: 16px;
    border: none;
}

/* File uploader */
.stFileUploader {
    border: 2px dashed #90CAF9;
    padding: 15px;
    border-radius: 10px;
}

/* Info Boxen */
div[data-testid="stInfo"] {
    background-color: #E3F2FD;
    border-radius: 10px;
    color: black !important;
}

/* Warning / Success Text */
div[data-testid="stWarning"],
div[data-testid="stSuccess"],
div[data-testid="stError"] {
    color: black !important;
}

/* Progress / captions */
.stCaption {
    color: black !important;
}

</style>
""", unsafe_allow_html=True)

# =============================
# SUPABASE
# =============================
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# =============================
# EINSTELLUNGEN
# =============================
HIGH_CONFIDENCE = 0.70
MID_CONFIDENCE = 0.50

# =============================
# HEADER
# =============================
st.title("🌿 RootWise")
st.subheader("Wildpflanzen scannen. Boden verstehen.")

st.markdown("### 📸 Wildpflanze scannen")
st.caption("Blatt und Blüte möglichst klar sichtbar fotografieren")

# =============================
# MODELL LADEN
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
    label = label.lower().strip()

    if "brennnessel" in label or "urtica" in label:
        return "brennnessel"
    if "löwenzahn" in label or "taraxacum" in label:
        return "loewenzahn"
    if "klee" in label or "trifolium" in label:
        return "klee"
    if "lamium" in label:
        return "taubnessel"
    if "schafgarbe" in label:
        return "schafgarbe"
    if "thymian" in label:
        return "thymian"
    if "kamille" in label:
        return "kamille"
    if "distel" in label:
        return "distel"
    if "farn" in label:
        return "farn"
    if "heidekraut" in label:
        return "heidekraut"

    return "unbekannt"

# =============================
# SUPABASE QUERY
# =============================
def get_plant_data(plant_key):
    res = supabase.table("plants") \
        .select("*") \
        .eq("plant_key", plant_key) \
        .execute()

    if res.data:
        return res.data[0]
    return None

# =============================
# UPLOAD
# =============================
uploaded_file = st.file_uploader(
    "📷 Pflanze hochladen",
    type=["jpg", "png", "jpeg"]
)

# =============================
# PROCESS
# =============================
if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except:
        st.error("❌ Bild konnte nicht geladen werden.")
        st.stop()

    col1, col2 = st.columns([1,1])

    with col1:
        st.image(image, caption="Dein Bild", use_column_width=True)

    # PREPROCESSING
    size = (224, 224)
    image_model = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image_model)
    normalized = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized

    # ANALYSE TEXT (bleibt wie gewünscht)
    with col2:
        st.markdown("### 🔍 Analyse läuft...")
        st.caption("Erkenne Wildpflanze → leite Boden ab → gebe Empfehlungen")

        with st.spinner("Analysiere Pflanze..."):
            prediction = model.predict(data)

    index = np.argmax(prediction)
    raw_label = class_names[index].strip()
    confidence = float(prediction[0][index])

    plant_key = normalize(raw_label)

    st.markdown("---")

    # MODEL INFO
    st.write(f"🔎 Modell erkennt: **{raw_label}**")
    st.progress(confidence)
    st.caption(f"Sicherheit: {round(confidence * 100, 1)} %")

    # =============================
    # HOHE SICHERHEIT
    # =============================
    if confidence >= HIGH_CONFIDENCE and plant_key != "unbekannt":

        st.success(f"🌿 Sicher erkannt: {plant_key}")

        plant_data = get_plant_data(plant_key)

        if plant_data:
            st.markdown("### 🌱 Bodenanalyse")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.info(f"🌍 Boden\n\n{plant_data['soil']}")
            with col2:
                st.info(f"💧 Feuchtigkeit\n\n{plant_data['moisture']}")
            with col3:
                st.info(f"☀️ Sonne\n\n{plant_data['sun']}")

            st.markdown("### 🌿 Empfehlungen")
            st.success(plant_data["recommendations"])

        else:
            st.warning("Keine Daten in Supabase gefunden.")

    # =============================
    # MITTLERE SICHERHEIT
    # =============================
    elif confidence >= MID_CONFIDENCE:

        st.warning("⚠️ Unsichere Erkennung – mögliche Pflanzen:")

        top_indices = np.argsort(prediction[0])[::-1][:3]

        options = []
        mapping = {}

        for i in top_indices:
            label = class_names[i].strip()
            conf = float(prediction[0][i])
            key = normalize(label)

            if key != "unbekannt":
                text = f"{key} ({round(conf*100, 1)}%)"
                options.append(text)
                mapping[text] = key

        if options:
            choice = st.selectbox("Welche Pflanze passt?", options)

            if st.button("🌱 Auswahl bestätigen & Analyse starten"):
                plant_key = mapping[choice]

                plant_data = get_plant_data(plant_key)

                if plant_data:
                    st.markdown("### 🌱 Bodenanalyse")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.info(f"🌍 Boden\n\n{plant_data['soil']}")
                    with col2:
                        st.info(f"💧 Feuchtigkeit\n\n{plant_data['moisture']}")
                    with col3:
                        st.info(f"☀️ Sonne\n\n{plant_data['sun']}")

                    st.markdown("### 🌿 Empfehlungen")
                    st.success(plant_data["recommendations"])

        else:
            st.error("Keine sinnvollen Vorschläge gefunden.")

    # =============================
    # NIEDRIGE SICHERHEIT (BLEIBT)
    # =============================
    else:
        st.error("❌ Zu unsicher erkannt (<50%)")
        st.info("Bitte besseres Bild aufnehmen (Licht, Nähe, Fokus)")

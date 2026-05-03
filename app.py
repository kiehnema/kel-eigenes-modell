import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
from supabase import create_client

# =============================
# CONFIG
# =============================
st.set_page_config(page_title="RootWise", layout="centered")

# =============================
# STYLE (Mobile App Look)
# =============================
st.markdown("""
<style>

/* Hintergrund */
body {
    background-color: #f4f7f6;
}

/* Zentrierte App */
.main {
    max-width: 420px;
    margin: auto;
    background: white;
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0 4px 25px rgba(0,0,0,0.05);
}

/* Buttons */
.stButton>button {
    background-color: #A8E6CF;
    color: black;
    border-radius: 14px;
    padding: 12px;
    font-size: 16px;
    border: none;
}

/* Upload Bereich */
.stFileUploader {
    border: 2px dashed #B8E0F5;
    padding: 15px;
    border-radius: 15px;
}

/* Rosa Empfehlung */
.recommendation-box {
    background-color: #FFD6E0;
    padding: 15px;
    border-radius: 15px;
    color: black;
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
st.markdown("## 🌿 RootWise")
st.caption("Wildpflanzen scannen. Boden verstehen.")

st.markdown("### 📸 Wildpflanze scannen")
st.caption("💡 Tipp: Blatt und Blüte gut sichtbar für beste Ergebnisse")

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
        return "Brennnessel"
    if "löwenzahn" in label or "taraxacum" in label:
        return "Löwenzahn"
    if "klee" in label or "trifolium" in label:
        return "Klee"
    if "lamium" in label:
        return "Taubnessel"
    if "schafgarbe" in label:
        return "Schafgarbe"
    if "thymian" in label:
        return "Thymian"
    if "kamille" in label:
        return "Kamille"
    if "distel" in label:
        return "Distel"
    if "farn" in label:
        return "Farn"
    if "heidekraut" in label:
        return "Heidekraut"

    return "Unbekannt"

# =============================
# SUPABASE QUERY
# =============================
def get_plant_data(plant_key):
    res = supabase.table("plants") \
        .select("*") \
        .eq("plant_key", plant_key.lower()) \
        .execute()

    if res.data:
        return res.data[0]
    return None

# =============================
# UPLOAD
# =============================
uploaded_file = st.file_uploader(
    "📷 Foto aufnehmen oder hochladen",
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

    st.image(image, caption="Deine Pflanze", use_column_width=True)

    # Preprocessing
    size = (224, 224)
    image_model = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image_model)
    normalized = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized

    # Prediction
    st.markdown("### 🔍 Analyse läuft...")
    st.caption("Die Pflanze wird erkannt und dein Boden daraus abgeleitet")

    with st.spinner("Analysiere..."):
        prediction = model.predict(data)

    index = np.argmax(prediction)
    raw_label = class_names[index].strip()
    confidence = float(prediction[0][index])

    plant_key = normalize(raw_label)

    # Confidence Anzeige
    st.progress(confidence)
    st.caption(f"Modellsicherheit: {round(confidence*100,1)}%")

    # =============================
    # HOHE SICHERHEIT
    # =============================
    if confidence >= HIGH_CONFIDENCE and plant_key != "Unbekannt":

        st.success(f"🌿 Erkannte Wildpflanze: **{plant_key}**")

        plant_data = get_plant_data(plant_key)

        if plant_data:

            # Bodenprofil
            st.markdown("### 🌱 Bodenprofil")

            col1, col2 = st.columns(2)

            with col1:
                st.info(f"🌍 Boden\n{plant_data['soil']}")
                st.info(f"💧 Feuchtigkeit\n{plant_data['moisture']}")

            with col2:
                st.info(f"☀️ Sonne\n{plant_data['sun']}")

            # Empfehlungen
            st.markdown("### 🌿 Empfehlungen")

            st.markdown(f"""
            <div class="recommendation-box">
            {plant_data["recommendations"]}
            </div>
            """, unsafe_allow_html=True)

        else:
            st.warning("Keine Daten zur Pflanze gefunden.")

    # =============================
    # MITTLERE SICHERHEIT
    # =============================
    elif confidence >= MID_CONFIDENCE:

        st.warning("⚠️ Unsichere Erkennung – bitte auswählen:")

        top_indices = np.argsort(prediction[0])[::-1][:3]

        options = []
        mapping = {}

        for i in top_indices:
            label = class_names[i].strip()
            conf = float(prediction[0][i])
            key = normalize(label)

            if key != "Unbekannt":
                text = f"{key} ({round(conf*100,1)}%)"
                options.append(text)
                mapping[text] = key

        if options:
            choice = st.selectbox("Welche Pflanze passt am besten?", options)

            if st.button("🌱 Auswahl bestätigen"):

                plant_key = mapping[choice]
                plant_data = get_plant_data(plant_key)

                if plant_data:

                    st.success(f"🌿 Gewählte Pflanze: {plant_key}")

                    st.markdown("### 🌱 Bodenprofil")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.info(f"🌍 Boden\n{plant_data['soil']}")
                        st.info(f"💧 Feuchtigkeit\n{plant_data['moisture']}")

                    with col2:
                        st.info(f"☀️ Sonne\n{plant_data['sun']}")

                    st.markdown("### 🌿 Empfehlungen")

                    st.markdown(f"""
                    <div class="recommendation-box">
                    {plant_data["recommendations"]}
                    </div>
                    """, unsafe_allow_html=True)

        else:
            st.error("Keine sinnvollen Vorschläge gefunden.")

    # =============================
    # NIEDRIGE SICHERHEIT
    # =============================
    else:
        st.error("❌ Erkennung zu unsicher (<50%)")
        st.info("Bitte besseres Bild aufnehmen (Licht, Fokus, Nähe)")

    # =============================
    # RESET BUTTON
    # =============================
    if st.button("🔄 Neue Pflanze scannen"):
        st.rerun()

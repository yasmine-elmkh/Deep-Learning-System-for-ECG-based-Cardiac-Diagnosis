# === 1. Couche d'attention personnalisée ===
from keras.saving import register_keras_serializable
from keras.layers import Layer
import tensorflow as tf

@register_keras_serializable()
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros", trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = tf.keras.activations.tanh(tf.linalg.matmul(x, self.W) + self.b)
        e = tf.squeeze(e, axis=-1)
        alpha = tf.keras.activations.softmax(e)
        alpha = tf.expand_dims(alpha, axis=-1)
        context = x * alpha
        context = tf.reduce_sum(context, axis=1)
        return context


# === 2. Chargement du modèle et du scaler ===
from tensorflow.keras.models import load_model
import joblib

model = load_model("ECG_model.h5", custom_objects={'Attention': Attention})
scaler = joblib.load("ECG_scaler.gz")


# === 3. Import des modules ===
import streamlit as st
import pandas as pd
import numpy as np
import io
from fpdf import FPDF


# === 4. Dictionnaires ===
classes = {
    0: "Normal : ECG normal sans anomalie.",
    1: "RBB : Bloc de branche droit.",
    2: "LBB : Bloc de branche gauche.",
    3: "PVC : Contraction ventriculaire prématurée.",
    4: "Autre : Autres anomalies cardiaques."
}

recommendations = {
    0: "Maintenez un mode de vie sain avec une activité physique régulière.",
    1: "Consultez un cardiologue pour évaluer le bloc de branche droit.",
    2: "Une évaluation approfondie par un professionnel est recommandée.",
    3: "Réduisez le stress et surveillez votre fréquence cardiaque.",
    4: "Consultez un professionnel pour un diagnostic approfondi."
}


# === 5. Fonction PDF avec nom complet ===
def generate_pdf_report(patient_id, nom_complet, pred, description, recommendation):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Rapport de prédiction ECG", ln=True, align="C")
    pdf.ln(10)
    pdf.cell(0, 10, f"Patient ID : {patient_id}", ln=True)
    pdf.cell(0, 10, f"Nom : {nom_complet}", ln=True)
    pdf.cell(0, 10, f"Classe prédite : {pred}", ln=True)
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"Description : {description}")
    pdf.ln(3)
    pdf.multi_cell(0, 10, f"Recommandation : {recommendation}")

    pdf_bytes = pdf.output(dest='S')
    if isinstance(pdf_bytes, str):
        pdf_bytes = pdf_bytes.encode('latin1')

    st.download_button(
        label="📄 Télécharger le rapport PDF",
        data=io.BytesIO(pdf_bytes),
        file_name=f"rapport_patient_{patient_id}.pdf",
        mime="application/pdf"
    )


# === 6. Interface Streamlit ===
st.title("🩺 Prédiction ECG – Diagnostic Cardiaque par Deep Learning")

# Champs de nom et prénom
nom_complet = st.text_input("🧍‍♂️ Nom complet du patient")

uploaded_file = st.file_uploader("📤 Charger un fichier ECG (.csv)", type=["csv"])

if uploaded_file is not None:
    if not nom_complet :
        st.warning("❗ Veuillez saisir le prénom et le nom du patient avant de continuer.")
    else:
        try:
            data = pd.read_csv(uploaded_file, header=None)

            if data.shape[1] != 187:
                st.error("⚠️ Le fichier doit contenir exactement 187 colonnes représentant le signal ECG.")
            else:
                ecg = scaler.transform(data)
                preds = model.predict(ecg)
                predictions = np.argmax(preds, axis=1)

                nom_complet = f"{nom_complet.strip().title()}"

                for i, pred in enumerate(predictions):
                    description = classes[pred]
                    recommendation = recommendations[pred]

                    st.success(f"✅ Patient {i+1} – Classe prédite : **{pred}**")
                    st.info(f"🩺 **{description}**")
                    st.markdown(f"🧠 **Suggestion personnalisée** : {recommendation}")

                    generate_pdf_report(
                        patient_id=i+1,
                        nom_complet=nom_complet,
                        pred=pred,
                        description=description,
                        recommendation=recommendation
                    )

        except Exception as e:
            st.error(f"❌ Erreur : {e}")

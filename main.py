import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Judul aplikasi
st.title("Prediksi Tingkat Keparahan Bencana Alam")

# Upload file Excel
uploaded_file = st.file_uploader("Upload File Excel (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    # Baca file Excel
    df = pd.read_excel(uploaded_file)

    # Hapus kolom tidak penting jika ada
    df = df.drop(columns=["Kronologi & Dokumentasi"], errors="ignore")

    # Tampilkan data
    st.subheader("Data Bencana")
    st.dataframe(df.head())

    # Bersihkan dan buat label target
    df = df.dropna(subset=["Kejadian", "Provinsi", "Rumah Rusak"])
    df["Rumah Rusak"] = pd.to_numeric(df["Rumah Rusak"], errors="coerce")

    def label_keparahan(x):
        if x < 10:
            return "Ringan"
        elif x <= 30:
            return "Sedang"
        else:
            return "Berat"

    df["Tingkat Keparahan"] = df["Rumah Rusak"].apply(label_keparahan)

    # Encode fitur
    le_kejadian = LabelEncoder()
    le_provinsi = LabelEncoder()
    df["Kejadian_encoded"] = le_kejadian.fit_transform(df["Kejadian"])
    df["Provinsi_encoded"] = le_provinsi.fit_transform(df["Provinsi"])

    X = df[["Kejadian_encoded", "Provinsi_encoded"]]
    y = df["Tingkat Keparahan"]

    # Latih model
    model = RandomForestClassifier()
    model.fit(X, y)

    st.subheader("Prediksi Tingkat Keparahan")

    # Input user
    kejadian_input = st.selectbox("Jenis Kejadian", df["Kejadian"].unique())
    provinsi_input = st.selectbox("Provinsi", df["Provinsi"].unique())

    kejadian_encoded = le_kejadian.transform([kejadian_input])[0]
    provinsi_encoded = le_provinsi.transform([provinsi_input])[0]

    pred = model.predict([[kejadian_encoded, provinsi_encoded]])[0]

    st.success(f"Prediksi Tingkat Keparahan: {pred}")

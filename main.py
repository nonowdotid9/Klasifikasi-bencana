import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import requests

st.set_page_config(page_title="Klasifikasi Bencana", layout="wide")

st.markdown("""
# 📊 Klasifikasi Tingkat Keparahan Bencana
Gunakan model machine learning untuk memprediksi dan memvisualisasikan tingkat keparahan bencana berdasarkan data historis.
""")

uploaded_file = st.file_uploader("📁 Upload File Excel (.xlsx)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file, header=1)
    df.columns = df.columns.str.strip()

    with st.expander("🔍 Lihat Data Mentah"):
        st.dataframe(df.head(), use_container_width=True)

    try:
        df = df.drop(columns=["Kronologi & Dokumentasi"], errors="ignore")
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

        le_kejadian = LabelEncoder()
        le_provinsi = LabelEncoder()
        df["Kejadian_encoded"] = le_kejadian.fit_transform(df["Kejadian"])
        df["Provinsi_encoded"] = le_provinsi.fit_transform(df["Provinsi"])

        X = df[["Kejadian_encoded", "Provinsi_encoded"]]
        y = df["Tingkat Keparahan"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        st.subheader("⚙️ Pilih Model Klasifikasi")
        selected_model = st.selectbox("Model yang ingin digunakan", ("Decision Tree", "Random Forest", "Logistic Regression", "Naive Bayes"))

        if selected_model == "Decision Tree":
            model = DecisionTreeClassifier()
        elif selected_model == "Random Forest":
            model = RandomForestClassifier()
        elif selected_model == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        else:
            model = GaussianNB()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Layout responsif
        col_left, col_right = st.columns([2, 3])

        with col_left:
            st.markdown(f"### 📈 Evaluasi Model - {selected_model}")
            report = classification_report(y_test, y_pred, output_dict=True)
            eval_df = pd.DataFrame(report).transpose()
            st.dataframe(eval_df, use_container_width=True)

            csv = eval_df.to_csv(index=True).encode('utf-8')
            st.download_button(
                label="⬇️ Download Hasil Evaluasi (CSV)",
                data=csv,
                file_name='evaluasi_model.csv',
                mime='text/csv',
                use_container_width=True
            )

            cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
            fig_cm = px.imshow(cm, 
                            labels=dict(x="Predicted", y="Actual", color="Jumlah"),
                            x=model.classes_, 
                            y=model.classes_,
                            color_continuous_scale="Blues",
                            title="Confusion Matrix")
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_right:
            st.markdown("### 🔮 Prediksi Interaktif")
            col1, col2 = st.columns(2)
            with col1:
                user_kejadian = st.selectbox("Jenis Kejadian", df["Kejadian"].unique())
            with col2:
                user_provinsi = st.selectbox("Provinsi", df["Provinsi"].unique())

            encoded_kejadian = le_kejadian.transform([user_kejadian])[0]
            encoded_provinsi = le_provinsi.transform([user_provinsi])[0]

            pred = model.predict([[encoded_kejadian, encoded_provinsi]])[0]

            st.success(f"📌 Prediksi Tingkat Keparahan: {pred}")

            pred_df = pd.DataFrame({
                'Jenis Kejadian': [user_kejadian],
                'Provinsi': [user_provinsi],
                'Prediksi Keparahan': [pred]
            })
            pred_csv = pred_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Hasil Prediksi", 
                data=pred_csv,
                file_name='hasil_prediksi.csv',
                mime='text/csv',
                use_container_width=True
            )

            st.markdown("### 📊 Visualisasi Fitur")
            if selected_model == "Logistic Regression":
                coef_df = pd.DataFrame({
                    'Fitur': X.columns,
                    'Koefisien': model.coef_[0]
                })
                fig = px.bar(coef_df, x="Koefisien", y="Fitur", orientation="h",
                            color="Koefisien", color_continuous_scale="RdBu",
                            title="Koefisien Logistic Regression")
                st.plotly_chart(fig, use_container_width=True)
            elif selected_model in ["Random Forest", "Decision Tree"]:
                feat_imp = pd.Series(model.feature_importances_, index=X.columns)
                fig = px.bar(feat_imp.sort_values(), orientation='h',
                            labels={'value': 'Importance', 'index': 'Fitur'},
                            title=f"Pentingnya Fitur - {selected_model}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Model ini tidak mendukung visualisasi fitur secara langsung.")

        st.markdown("### 🗺️ Visualisasi Peta Tingkat Keparahan Bencana per Provinsi")
        severity_map = {'Ringan': 1, 'Sedang': 2, 'Berat': 3}
        df['Severity_Score'] = df['Tingkat Keparahan'].map(severity_map)
        avg_severity = df.groupby('Provinsi')["Severity_Score"].mean().reset_index()
        avg_severity['Provinsi'] = avg_severity['Provinsi'].str.upper().str.strip()

        try:
            geojson_url = 'https://raw.githubusercontent.com/superpikar/indonesia-geojson/master/indonesia-province.json'
            response = requests.get(geojson_url)
            if response.status_code == 200:
                geojson = response.json()
                fig = px.choropleth(
                    avg_severity,
                    geojson=geojson,
                    featureidkey='properties.Propinsi',
                    locations='Provinsi',
                    color='Severity_Score',
                    color_continuous_scale=['green', 'yellow', 'red'],
                    range_color=(1, 3),
                    labels={'Severity_Score': 'Keparahan Rata-rata'},
                    title='Peta Tingkat Keparahan Bencana per Provinsi'
                )
                fig.update_geos(fitbounds='locations', visible=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Gagal ambil GeoJSON. Status code:", response.status_code)
        except Exception as e:
            st.warning(f"Gagal memuat peta: {e}")

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat memproses: {e}")

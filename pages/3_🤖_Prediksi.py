import streamlit as st
import pandas as pd
import joblib

st.title("🤖 Prediksi Status Gizi")

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

umur = st.number_input("Umur (bulan)", 0, 240, 24)
jk = st.radio("Jenis Kelamin", ["Laki-laki", "Perempuan"])
berat = st.number_input("Berat", 0.0, 50.0, 10.0)
tinggi = st.number_input("Tinggi", 0.0, 150.0, 80.0)
lila = st.number_input("LiLA", 0.0, 30.0, 14.0)

jk_encoded = 1 if jk == "Laki-laki" else 2

# mapping label
label_mapping = {
    0: "Gizi Normal/baik",
    1: "Gizi Kurang",
    2: "Gizi Buruk",
    3: "Gizi Lebih",
    4: "Resiko gizi lebih",
    5: "Obesitas"
}

if st.button("Prediksi"):
    df = pd.DataFrame([{
        'umur(bln)': umur,
        'encoding JK': jk_encoded,
        'Berat': berat,
        'Tinggi': tinggi,
        'LiLA': lila
    }])

    # scaling
    kolom_scale = ['umur(bln)', 'Berat', 'Tinggi', 'LiLA']
    scaled = scaler.transform(df[kolom_scale])

    df_scaled = pd.concat([
        pd.DataFrame(scaled, columns=kolom_scale),
        df[['encoding JK']]
    ], axis=1)

    # 🔥 WAJIB: urutkan ulang kolom
    df_scaled = df_scaled[['umur(bln)', 'encoding JK', 'Berat', 'Tinggi', 'LiLA']]

    pred = model.predict(df_scaled)[0]
    nama_kelas = label_mapping[pred]

    st.success(f"Hasil: {pred} - {nama_kelas}")
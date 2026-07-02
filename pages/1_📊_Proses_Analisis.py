import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.title("📊 Proses Analisis")

tab1, tab2, tab3, tab4 = st.tabs([
    "Load Dataset",
    "Preprocessing",
    "Outlier",
    "Splitting Data"  
])

# =====================================================================================
# LOAD DATASET
df1 = pd.read_excel("DATA DESA SIMOGIRANG.xlsx")
# Simogirang
X1 = df1[['Tgl Lahir Manual', 'Jenis Kelamin', 'Berat Lahir', 'Panjang Badan Lahir', 'LILA (Cm)']]
y1 = df1['Unnamed: 34']
df_simogirang = pd.concat([X1, y1], axis=1)
df_simogirang = df_simogirang.rename(columns={
    'Tgl Lahir Manual': 'Tgl Lahir',
    'Jenis Kelamin': 'JK',
    'Berat Lahir': 'Berat',
    'Panjang Badan Lahir': 'Tinggi',
    'LILA (Cm)': 'LiLA',
    'Unnamed: 34': 'Status Gizi'
})
df_simogirang = df_simogirang.drop([0,1]).reset_index(drop=True)

# Kaggle
df2 = pd.read_excel("datakaggle.xlsx")
X2 = df2[['Tgl Lahir', 'JK', 'Berat', 'Tinggi', 'LiLA']]
y2 = df2['Status Gizi']
df_kaggle = pd.concat([X2, y2], axis=1)

# Gabungan
df_gabungan = pd.concat([df_simogirang, df_kaggle], ignore_index=True)
with tab1:
    st.header("📂 Load Dataset Simogirang")
    st.dataframe(df1)

    st.subheader("Subset Simogirang")
    st.dataframe(df_simogirang)

    st.header("📂 Load Dataset Kaggle")
    st.dataframe(df2)

    st.subheader("Subset Kaggle")
    st.dataframe(df_kaggle)

    st.header("📂 Dataset Gabungan")
    st.dataframe(df_gabungan)
# ================================================================================

with tab2:
    st.header("🔧 Preprocessing Dataset")
    st.markdown("### 1. Cek Tipe Data")

    # 1. Convert Tgl Lahir ke datetime
    df_gabungan['Tgl Lahir'] = pd.to_datetime(df_gabungan['Tgl Lahir'], errors='coerce')

    # 2. Pastikan JK dan status_gizi bertipe string/object
    df_gabungan['JK'] = df_gabungan['JK'].astype('string')
    df_gabungan['Status Gizi'] = df_gabungan['Status Gizi'].astype('object')

    # 3. Convert kolom numerik
    numeric_cols = ['Berat', 'Tinggi', 'LiLA']
    for col in numeric_cols:
        df_gabungan[col] = pd.to_numeric(df_gabungan[col], errors='coerce')

    # TAMPILKAN INFO DATAFRAME
    import io
    buffer = io.StringIO()
    df_gabungan.info(buf=buffer)
    info_str = buffer.getvalue()
    # st.subheader("Informasi Dataset Gabungan")
    st.text(info_str)
# ====================================================================================
    st.markdown("### 2. Penanganan Missing Value")
    # jumlah missing value sebelum preprocessing
    missing_sebelum = df_gabungan.isnull().sum()
    # ubah nilai 0 menjadi NaN
    df_gabungan[['Berat', 'Tinggi', 'LiLA']] = (
        df_gabungan[['Berat', 'Tinggi', 'LiLA']].replace(0, np.nan)
    )
    # isi missing value dengan median
    median_cols = ['Berat', 'Tinggi', 'LiLA']
    for col in median_cols:
        df_gabungan[col] = df_gabungan[col].fillna(
            df_gabungan[col].median()
        )
    # hapus missing value pada kolom tertentu
    df_gabungan = df_gabungan.dropna(
        subset=['Tgl Lahir', 'JK', 'Status Gizi']
    )
    # jumlah missing value setelah preprocessing
    missing_sesudah = df_gabungan.isnull().sum()
    # gabungkan menjadi tabel
    tabel_missing = pd.DataFrame({
        'Nama Fitur': missing_sebelum.index,
        'Missing Value Sebelum': missing_sebelum.values,
        'Missing Value Sesudah': missing_sesudah.values
    })
    # tampilkan tabel
    st.dataframe(tabel_missing)
# ====================================================================================
    st.markdown("### 3. Penanganan Data Duplikat")
    jumlah_sebelum = len(df_gabungan)
    df_gabungan = df_gabungan.drop_duplicates()
    jumlah_sesudah = len(df_gabungan)

    # tampilkan informasi
    st.write(f"Jumlah data sebelum hapus duplikat: {jumlah_sebelum}")
    st.write(f"Jumlah data setelah hapus duplikat: {jumlah_sesudah}")
    st.write(f"Jumlah data yang dihapus: {jumlah_sebelum - jumlah_sesudah}")
    # tampilkan data setelah hapus duplikat
    st.subheader("Data Setelah Hapus Duplikat")
    st.dataframe(df_gabungan)
# ====================================================================================
    st.markdown("### 4. Perbaikan Label Status Gizi")
    # Tampilkan status gizi sebelum di-replace
    st.subheader("Sebelum Perbaikan Label")
    st.dataframe(df_gabungan['Status Gizi'].value_counts())
    # Mapping perubahan label
    mapping_status = {
        'Normal': 'Normal',
        'Gz.Krg': 'Gizi Kurang',
        'Gz.Brk': 'Gizi Buruk',
        'Gizi Baik': 'Gizi Baik',
        'Gz.Lbh': 'Gizi Lebih',
        'R.Gz.Lbh': 'Resiko Gizi Lebih',
        'Obesitas': 'Obesitas',
    }
    # gabungkan Normal & Gizi Baik
    df_gabungan['Status Gizi'] = df_gabungan['Status Gizi'].replace({
        'Normal': 'Gizi Normal/Baik',
        'Gizi Baik': 'Gizi Normal/Baik'
    })
    # Terapkan replace
    df_gabungan['Status Gizi'] = df_gabungan['Status Gizi'].replace(mapping_status)
    # Tampilkan setelah replace
    st.subheader("Setelah Perbaikan Label")
    st.dataframe(df_gabungan['Status Gizi'].value_counts())
# ===================================================================================
    st.markdown("### 5. Encoding Dataset")
    # convert Tgl Lahir ke umur(bln)
    # today = pd.to_datetime('today')
    today = pd.Timestamp('2026-05-11')
    df_gabungan['umur(bln)'] = ((today - df_gabungan['Tgl Lahir']).dt.days // 30).astype(int)
    # encoding JK
    df_gabungan['JK'] = df_gabungan['JK'].astype(str).str.upper().str.strip()
    mapping_jk = {'L': 1, 'P': 2}
    df_gabungan['encoding JK'] = df_gabungan['JK'].map(mapping_jk)
    # mapping status gizi
    mapping_gizi = {
        'Gizi Normal/Baik': 0,
        'Gizi Kurang': 1,
        'Gizi Buruk': 2,
        'Gizi Lebih': 3,
        'Resiko Gizi Lebih': 4,
        'Obesitas': 5
    }
    df_gabungan['encoding status gizi'] = df_gabungan['Status Gizi'].map(mapping_gizi)
    st.dataframe(df_gabungan[['Tgl Lahir', 'umur(bln)', 'JK', 'encoding JK', 'Status Gizi', 'encoding status gizi']])

# ===============================================================================
# PROSES OUTLIER
numerik = ['Berat', 'Tinggi', 'umur(bln)', 'LiLA']
# copy biar aman
df_outlier = df_gabungan.copy()
# hapus outlier LiLA
df_outlier = df_outlier[~df_outlier['LiLA'].isin([43, 46, 47])]
with tab3:
    # st.header("Analisis & Penanganan Outlier")
    # HISTOGRAM SEBELUM OUTLIER
    st.header("Histogram Sebelum Hapus Outlier")
    fig, ax = plt.subplots(2, 2, figsize=(10,6))

    for i, col in enumerate(numerik):
        r = i // 2
        c = i % 2

        ax[r, c].hist(df_gabungan[col], bins=20)
        ax[r, c].set_title(f"Histogram {col}")
        ax[r, c].set_xlabel(col)
        ax[r, c].set_ylabel("Frekuensi")

    plt.tight_layout()
    st.pyplot(fig)

    # HISTOGRAM LIla setelah outlier
    st.header("Histogram LiLA Setelah Hapus Outlier")

    fig2, ax2 = plt.subplots()

    ax2.hist(df_outlier['LiLA'], bins=20)
    ax2.set_title("Histogram LiLA Setelah Hapus Outlier")

    st.pyplot(fig2)
# ===============================================================================
# ===============================
# SPLITTING (DI LUAR TAB)
# ===============================

fitur_pilih = ['umur(bln)', 'encoding JK', 'Berat', 'Tinggi', 'LiLA']
target = 'encoding status gizi'

# ⚠️ PAKAI DATA SETELAH OUTLIER
X = df_outlier[fitur_pilih]
y = df_outlier[target]

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# SCALING
# ===============================
kolom_scale = ['umur(bln)', 'Berat', 'Tinggi', 'LiLA']
kolom_tidak_scale = ['encoding JK']

urutan_kolom = ['umur(bln)', 'encoding JK', 'Berat', 'Tinggi', 'LiLA']

# scaler_train = MinMaxScaler()
scaler_train = StandardScaler()
scaler_train.fit(X_train[kolom_scale])

X_train_scaled_part = scaler_train.transform(X_train[kolom_scale])
X_test_scaled_part = scaler_train.transform(X_test[kolom_scale])

X_train_scaled = pd.concat([
    pd.DataFrame(X_train_scaled_part, columns=kolom_scale, index=X_train.index),
    X_train[kolom_tidak_scale]
], axis=1)

X_test_scaled = pd.concat([
    pd.DataFrame(X_test_scaled_part, columns=kolom_scale, index=X_test.index),
    X_test[kolom_tidak_scale]
], axis=1)

X_train_scaled = X_train_scaled[urutan_kolom]
X_test_scaled = X_test_scaled[urutan_kolom]
with tab4:
    st.header("Splitting dataset")

    st.subheader("📊 Pembagian Dataset (Train/Test)")
    st.write(f"Jumlah data Train : {len(X_train)}")
    st.write(f"Jumlah data Test  : {len(X_test)}")

    st.markdown("### Normalisasi dataset")
    st.write(X_train_scaled)

# ===============================
# SIMPAN KE SESSION STATE
# ===============================
st.session_state.X_train = X_train
st.session_state.X_test = X_test
st.session_state.y_train = y_train
st.session_state.y_test = y_test
st.session_state.X_train_scaled = X_train_scaled
st.session_state.X_test_scaled = X_test_scaled
st.session_state.scaler = scaler_train
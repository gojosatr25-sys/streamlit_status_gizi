import streamlit as st

st.set_page_config(layout="wide")

st.markdown("""
    <style>
    .centered {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 80vh;
        text-align: center;
    }
    .title {
        font-size: 60px;
        font-weight: bold;
        color: #2c3e50;
    }
    .subtitle {
        font-size: 30px;
        color: #7f8c8d;
        margin-top: 10px;
    }
    </style>

    <div class="centered">
        <div class="title">Aplikasi Status Gizi Anak 👶</div>
        <div class="subtitle">
            Sistem prediksi status gizi berdasarkan data antropometri anak
        </div>
    </div>
""", unsafe_allow_html=True)
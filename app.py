import streamlit as st
import pandas as pd
import joblib

# 1. PERSIAPAN MUAT ARTEFAK DENGAN CACHE
@st.cache_resource
def load_artefak():
    scaler = joblib.load("scaler_credit_card.joblib")
    model_kmeans = joblib.load("kmeans_credit_card.joblib")
    return scaler, model_kmeans

scaler, model_kmeans = load_artefak()

# 2. UI STREAMLIT
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Prediksi Cluster Pelanggan Kartu Kredit")
st.write(
    "Aplikasi ini digunakan untuk mengelompokkan pelanggan "
    "berdasarkan pola penggunaan kartu kredit."
)

st.markdown("---")
st.subheader("Input Data Pelanggan")

col1, col2 = st.columns(2)

with col1:
    balance_val = st.number_input("BALANCE", min_value=0.0, value=1000.0)
    purchases_val = st.number_input("PURCHASES", min_value=0.0, value=500.0)
    cash_advance_val = st.number_input("CASH_ADVANCE", min_value=0.0, value=200.0)

with col2:
    credit_limit_val = st.number_input("CREDIT_LIMIT", min_value=0.0, value=3000.0)
    payments_val = st.number_input("PAYMENTS", min_value=0.0, value=500.0)
    purchases_trx_val = st.number_input("PURCHASES_TRX", min_value=0, value=10)

# 3. LOGIKA PREDIKSI
if st.button("Prediksi Cluster", use_container_width=True):

    input_data = pd.DataFrame(
        [[
            balance_val,
            purchases_val,
            cash_advance_val,
            credit_limit_val,
            payments_val,
            purchases_trx_val
        ]],
        columns=[
            "BALANCE",
            "PURCHASES",
            "CASH_ADVANCE",
            "CREDIT_LIMIT",
            "PAYMENTS",
            "PURCHASES_TRX"
        ]
    )

    # Standardisasi menggunakan scaler yang sama saat training
    scaled_input = scaler.transform(input_data)

    # Prediksi cluster menggunakan model K-Means
    cluster_result = model_kmeans.predict(scaled_input)[0]

    st.success(
        f"Pelanggan ini masuk ke dalam **Cluster {cluster_result}**"
    )

    st.metric(
        label="Hasil Pengelompokan",
        value=f"Cluster {cluster_result}"
    )

    st.info(
        "Cluster merupakan hasil pengelompokan berdasarkan kemiripan "
        "pola penggunaan kartu kredit pada fitur yang digunakan."
    )

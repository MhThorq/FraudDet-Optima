import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load Pipeline Logic dari Notebook Anda
class FraudInferencePipeline:
    def __init__(self):
        self.xgb_model = joblib.load('fraud_model_xgboost.pkl')
        self.iso_model = joblib.load('iso_forest_model.pkl')
        self.encoders = joblib.load('label_encoders.pkl')
        self.state = joblib.load('state_mapping.pkl')
        self.cat_cols = joblib.load('cat_cols.pkl')
        self.num_cols = joblib.load('num_cols.pkl')
        self.threshold = 0.3

    def preprocess(self, raw_data):
        # 1. Konversi input ke DataFrame
        df = pd.DataFrame([raw_data])

        # 2. Reindex agar memiliki kolom asli (cat + num)
        # Pastikan kolom isFraud tidak ikut karena itu target
        base_features = [c for c in list(self.cat_cols) + list(self.num_cols) if c != 'isFraud']
        df = df.reindex(columns=base_features)

        # 3. Imputasi dasar
        df[self.cat_cols] = df[self.cat_cols].fillna('MISSING')
        df[self.num_cols] = df[self.num_cols].fillna(-999)

        # 4. Label Encoding
        for col in self.cat_cols:
            le = self.encoders[col]
            df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

        # --- LANGKAH KRUSIAL: ANOMALY SCORE ---
        # Kita ambil anomaly score menggunakan kolom asli SAJA (sesuai saat fit)
        # Drop TransactionID jika ada karena biasanya tidak dipakai saat fit IsoForest
        features_for_iso = df.drop(columns=['TransactionID'], errors='ignore')
        df['anomaly_score'] = self.iso_model.predict(features_for_iso)
        # --------------------------------------

        # 5. BARU TAMBAHKAN Feature Engineering (Agregat & Time)
        # Menambahkan fitur setelah anomaly score agar tidak merusak input IsoForest
        df['Transaction_hour'] = (df['TransactionDT'] / 3600) % 24

        for col in ['card1', 'card2', 'card3', 'card5', 'addr1', 'P_emaildomain']:
            df[f'{col}_count'] = df[col].map(self.state['counts'].get(col, {})).fillna(0)

        card1_map = self.state['means'].get('card1_mean', {})
        df['Amt_to_mean_card1'] = df['TransactionAmt'] / df['card1'].map(card1_map).fillna(1)
        
        df['trans_count_per_hour'] = df.apply(
            lambda x: self.state['hourly'].get((x['card1'], x['Transaction_hour']), 0), axis=1
        )

        return df

    def predict(self, raw_data):
        processed_df = self.preprocess(raw_data)
        prob = self.xgb_model.predict_proba(processed_df)[:, 1][0]
        prediction = 1 if prob >= self.threshold else 0
        return {"is_fraud": bool(prediction), "confidence": round(float(prob), 4)}

# --- UI Streamlit ---
st.title("🛡️ Digital Payment Fraud Detection")
st.write("Masukkan detail transaksi untuk memeriksa risiko fraud.")

# Input Form (Sesuaikan dengan fitur utama Anda)
with st.form("transaction_form"):
    col1, col2 = st.columns(2)
    with col1:
        tx_id = st.number_input("Transaction ID", value=3000000)
        tx_amt = st.number_input("Transaction Amount", value=50.0)
        card1 = st.number_input("Card 1", value=13926)
    with col2:
        tx_dt = st.number_input("Transaction DT (Seconds)", value=86400)
        p_email = st.text_input("P_emaildomain", value="gmail.com")
        addr1 = st.number_input("Address 1", value=315.0)

    # Note: Tambahkan semua field yang diperlukan oleh model Anda di sini

    submitted = st.form_submit_button("Check Transaction")

if submitted:
    pipeline = FraudInferencePipeline()
    # Dummy data dictionary dari input
    data = {
        "TransactionID": tx_id,
        "TransactionAmt": tx_amt,
        "TransactionDT": tx_dt,
        "card1": card1,
        "P_emaildomain": p_email,
        "addr1": addr1,
        # ... pastikan semua kolom di raw_data terpenuhi
    }
    
    result = pipeline.predict(data)
    
    if result["is_fraud"]:
        st.error(f"🚨 HIGH RISK! Score: {result['confidence']}")
    else:
        st.success(f"✅ SAFE. Score: {result['confidence']}")
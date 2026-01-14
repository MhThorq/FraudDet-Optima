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
        df = pd.DataFrame([raw_data])
        df[self.cat_cols] = df[self.cat_cols].fillna('MISSING')
        df[self.num_cols] = df[self.num_cols].fillna(-999)

        for col in self.cat_cols:
            le = self.encoders[col]
            df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

        df['Transaction_hour'] = (df['TransactionDT'] / 3600) % 24

        for col in ['card1', 'card2', 'card3', 'card5', 'addr1', 'P_emaildomain']:
            df[f'{col}_count'] = df[col].map(self.state['counts'][col]).fillna(0)

        df['Amt_to_mean_card1'] = df['TransactionAmt'] / df['card1'].map(self.state['means']['card1_mean']).fillna(1)
        
        df['trans_count_per_hour'] = df.apply(
            lambda x: self.state['hourly'].get((x['card1'], x['Transaction_hour']), 0), axis=1
        )

        df['anomaly_score'] = self.iso_model.predict(df.drop(columns=['TransactionID'], errors='ignore'))
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
    
    submitted = st.form_submit_with_button("Check Transaction")

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
import streamlit as st
import pandas as pd
import joblib
import numpy as np


class FraudInferencePipeline:
    def __init__(self):
        self.xgb_model = joblib.load('fraud_model_xgboost.pkl')
        self.iso_model = joblib.load('iso_forest_model.pkl')
        self.encoders = joblib.load('label_encoders.pkl')
        self.state = joblib.load('state_mapping.pkl')
        self.cat_cols = joblib.load('cat_cols.pkl')
        self.num_cols = joblib.load('num_cols.pkl')
        self.threshold = 0.05

    def preprocess(self, raw_data):
        # 1. Konversi input ke DataFrame
        df = pd.DataFrame([raw_data])

        # 2. Reindex agar memiliki kolom asli (cat + num), tanpa isFraud
        base_features = [c for c in list(self.cat_cols) + list(self.num_cols) if c != 'isFraud']
        df = df.reindex(columns=base_features)

        # 3. Imputasi dasar
        cols_to_fill_cat = [c for c in self.cat_cols if c in df.columns]
        cols_to_fill_num = [c for c in self.num_cols if c in df.columns]

        df[cols_to_fill_cat] = df[cols_to_fill_cat].fillna('MISSING')
        df[cols_to_fill_num] = df[cols_to_fill_num].fillna(-999)

        # 4. Label Encoding
        for col in cols_to_fill_cat:
            le = self.encoders[col]
            df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

        # 5. Anomaly Score (IsolationForest)
        features_for_iso = df[self.iso_model.feature_names_in_]
        anomaly_score = self.iso_model.predict(features_for_iso)

        # 6. Feature Engineering — dikumpulkan dulu, baru pd.concat sekaligus
        transaction_hour = (df['TransactionDT'] / 3600) % 24

        count_cols = {}
        for col in ['card1', 'card2', 'card3', 'card5', 'addr1', 'P_emaildomain']:
            count_cols[f'{col}_count'] = df[col].map(self.state['counts'].get(col, {})).fillna(0)

        card1_map = self.state['means'].get('card1_mean', {})
        amt_to_mean_card1 = df['TransactionAmt'] / df['card1'].map(card1_map).fillna(1)

        trans_count_per_hour = df.apply(
            lambda x: self.state['hourly'].get((x['card1'], (x['TransactionDT'] / 3600) % 24), 0),
            axis=1
        )

        # Gabungkan semua kolom baru sekaligus (menghindari fragmentasi)
        new_cols = pd.DataFrame(
            {
                'anomaly_score': anomaly_score,
                'Transaction_hour': transaction_hour,
                **count_cols,
                'Amt_to_mean_card1': amt_to_mean_card1,
                'trans_count_per_hour': trans_count_per_hour,
            },
            index=df.index
        )
        df = pd.concat([df, new_cols], axis=1)

        # 7. Pastikan urutan kolom sesuai ekspektasi XGBoost
        df = df[self.xgb_model.feature_names_in_]

        return df

    def predict(self, raw_data):
        processed_df = self.preprocess(raw_data)
        prob = self.xgb_model.predict_proba(processed_df)[:, 1][0]
        prediction = 1 if prob >= self.threshold else 0
        return {"is_fraud": bool(prediction), "confidence": round(float(prob), 4)}


# --- UI Streamlit ---
st.title("🛡️ Digital Payment Fraud Detection")
st.write("Masukkan detail transaksi untuk memeriksa risiko fraud.")

with st.form("transaction_form"):
    col1, col2 = st.columns(2)
    with col1:
        tx_id = st.number_input("ID Transaksi", value=3000000)
        tx_amt = st.number_input("Jumlah Transaksi", value=50.0)
        card1 = st.number_input("Identitas Kartu", value=13926)
    with col2:
        tx_dt = st.number_input("Waktu Transaksi (Detik)", value=86400)
        p_email = st.text_input("Domain Email", value="gmail.com")
        addr1 = st.number_input("Kode Wilayah Transaksi", value=315.0)

    submitted = st.form_submit_button("Cek Transaksi")

if submitted:
    pipeline = FraudInferencePipeline()
    data = {
        "TransactionID": tx_id,
        "TransactionAmt": tx_amt,
        "TransactionDT": tx_dt,
        "card1": card1,
        "P_emaildomain": p_email,
        "addr1": addr1,
    }

    result = pipeline.predict(data)
    st.write(f"Raw Probability: {result['confidence']}")

    if result["is_fraud"]:
        st.error(f"🚨 HIGH RISK! Score: {result['confidence']}")
    else:
        st.success(f"✅ SAFE. Score: {result['confidence']}")

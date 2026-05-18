import time
import pandas as pd
import numpy as np
from app import FraudInferencePipeline # Pastikan file app.py berada di direktori yang sama

def run_batch_test(n_samples=1000):
    # 1. Inisialisasi Pipeline
    pipeline = FraudInferencePipeline()
    
    # 2. Siapkan Data Uji Sintetis (Batch)
    # Membuat 1000 data dummy berdasarkan fitur yang dibutuhkan model Anda
    batch_data = []
    for i in range(n_samples):
        batch_data.append({
            "TransactionID": 3000000 + i,
            "TransactionAmt": np.random.uniform(1, 5000),
            "TransactionDT": 86400 + i,
            "card1": np.random.randint(1000, 20000),
            "P_emaildomain": "gmail.com",
            "addr1": 315.0,
            "ProductCD": "W"
        })

    print(f"Memulai pengujian batch untuk {n_samples} transaksi...")

    # 3. Eksekusi Pengujian dan Pengukuran Waktu
    start_time = time.time()
    
    results = []
    for data in batch_data:
        res = pipeline.predict(data)
        results.append(res)
    
    end_time = time.time()

    # 4. Kalkulasi Metrik Performa
    total_duration = end_time - start_time
    avg_latency = (total_duration / n_samples) * 1000 # dalam milidetik
    throughput = n_samples / total_duration # transaksi per detik

    print("-" * 30)
    print(f"Hasil Pengujian Performa:")
    print(f"Total Waktu      : {total_duration:.4f} detik")
    print(f"Rata-rata Latensi: {avg_latency:.2f} ms per transaksi")
    print(f"Throughput       : {throughput:.2f} transaksi/detik")
    print("-" * 30)

if __name__ == "__main__":
    run_batch_test(1000) # Uji dengan 1000 transaksi    
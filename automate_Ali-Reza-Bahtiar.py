import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(input_file, output_file):
    """
    Fungsi untuk otomatisasi pembersihan data.
    """
    try:
        # 1. Load Data
        df = pd.read_csv(input_file)
        print(f"[*] Berhasil memuat {input_file}")

        # 2. Penanganan Missing Values
        # (Logika disamakan dengan notebook sebelumnya)
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
        
        # 3. Encoding Kategorikal
        categorical_cols = df.select_dtypes(include=['object']).columns
        le = LabelEncoder()
        for col in categorical_cols:
            df[col] = le.fit_transform(df[col])

        # 4. Scaling Numerik
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        # 5. Simpan Hasil
        df.to_csv(output_file, index=False)
        print(f"[V] Data bersih disimpan ke: {output_file}")
        
    except Exception as e:
        print(f"[X] Terjadi kesalahan: {e}")

if __name__ == "__main__":
    # Jalankan fungsi otomatisasi
    preprocess_data('dataset_heart.csv', 'dataset_preprocessing.csv')
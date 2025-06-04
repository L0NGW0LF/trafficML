import pandas as pd
import numpy as np
import joblib
import os
import sys

# === Input da linea di comando ===
if len(sys.argv) < 2:
    print("❌ Specificare il file CSV da analizzare. Uso: python script.py <percorso/file.csv>")
    sys.exit(1)

file_path = sys.argv[1]
file_base = os.path.splitext(os.path.basename(file_path))[0]

# === Setup percorsi output ===
output_dir = 'data/processed'
log_dir = 'out/logs'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

output_file = os.path.join(output_dir, f'{file_base}_transformed.csv')
attack_log_path = os.path.join(log_dir, f'{file_base}_attacks.csv')

# === Colonne ===
new_column_names = [
    "Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port", "Protocol", "Timestamp",
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Total Length of Fwd Packets",
    "Total Length of Bwd Packets", "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Fwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max",
    "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total",
    "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags",
    "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size",
    "Avg Bwd Segment Size", "Fwd Header Length",  
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min", "Label"
]
# === Fase 1: carica e modifica ===
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

if "Fwd Header Length" not in df.columns:
    raise ValueError("Colonna 'Fwd Header Length' non trovata nel dataset!")

fwd_header_values = df["Fwd Header Length"].copy()
insert_pos = df.columns.get_loc("Bwd Segment Size Avg") + 1
df.insert(insert_pos, "Fwd Header Length DUPLICATE", fwd_header_values)

if len(new_column_names) != df.shape[1]:
    raise ValueError(f"Numero colonne errato: aspettate {len(new_column_names)}, trovate {df.shape[1]}")

df.columns = new_column_names
df.to_csv(output_file, index=False)
print(f"✅ File trasformato e salvato: {output_file}")

# === Fase 2: predizione ===
model = joblib.load('out/model/rf_model_balanced.joblib')
label_encoder = joblib.load('out/model/label_encoder.joblib')
scaler = joblib.load('out/model/scaler.joblib')
imputer = joblib.load('out/model/imputer.joblib')
selected_features = joblib.load('out/model/selected_features.joblib')

raw_data = pd.read_csv(output_file)
raw_data.columns = raw_data.columns.str.strip()

meta_info = raw_data[['Source IP', 'Destination IP', 'Destination Port']].copy()
new_data = raw_data.drop(columns=['Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'Label'], errors='ignore')

for col in new_data.select_dtypes(include=['object']).columns:
    new_data[col] = new_data[col].astype(str).factorize()[0]

new_data.replace([np.inf, -np.inf], np.nan, inplace=True)
new_data = pd.DataFrame(imputer.transform(new_data), columns=imputer.feature_names_in_)
new_data = pd.DataFrame(scaler.transform(new_data), columns=scaler.feature_names_in_)
new_data = new_data[selected_features]

predicted_labels = label_encoder.inverse_transform(model.predict(new_data))

results = meta_info.copy()
results['Predicted Label'] = predicted_labels
attacks = results[results['Predicted Label'] != 'BENIGN']

if attacks.empty:
    print("✅ Nessun attacco rilevato.")
else:
    unique_attacks = attacks.drop_duplicates(subset=['Source IP', 'Destination IP', 'Destination Port', 'Predicted Label'])
    unique_attacks.to_csv(attack_log_path, index=False)
    
    print(f"🚨 {len(unique_attacks)} attacchi unici rilevati. Salvati in: {attack_log_path}")
    for _, row in unique_attacks.iterrows():
        print(f"🔐 {row['Source IP']} → {row['Destination IP']} :{row['Destination Port']} ({row['Predicted Label']})")
    print("✅ Analisi completata.")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import joblib
from sklearn.ensemble import RandomForestClassifier as skRF
import shap
import gc


def loader(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    df = df[df['Flow ID'] != 'Flow ID']
    return df

# Primo blocco
df1 = loader('/content/drive/MyDrive/TrafficLabelling/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
df2 = loader('/content/drive/MyDrive/TrafficLabelling/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv')
part1 = pd.concat([df1, df2], ignore_index=True)
del df1, df2
gc.collect()

# Secondo blocco
df3 = loader('/content/drive/MyDrive/TrafficLabelling/Friday-WorkingHours-Morning.pcap_ISCX.csv')
df4 = loader('/content/drive/MyDrive/TrafficLabelling/Monday-WorkingHours.pcap_ISCX.csv')
part2 = pd.concat([df3, df4], ignore_index=True)
del df3, df4
gc.collect()

# Terzo blocco
df5 = loader('/content/drive/MyDrive/TrafficLabelling/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv')
df6 = pd.read_csv('/content/drive/MyDrive/TrafficLabelling/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv', low_memory=False, encoding='latin-1')
df6.columns = df6.columns.str.strip()
df6 = df6[df6['Flow ID'] != 'Flow ID']
part3 = pd.concat([df5, df6], ignore_index=True)
del df5, df6
gc.collect()

# Quarto blocco
df7 = loader('/content/drive/MyDrive/TrafficLabelling/Tuesday-WorkingHours.pcap_ISCX.csv')
df8 = loader('/content/drive/MyDrive/TrafficLabelling/Wednesday-workingHours.pcap_ISCX.csv')
part4 = pd.concat([df7, df8], ignore_index=True)
del df7, df8
gc.collect()

# Concatenazione finale
df = pd.concat([part1, part2, part3, part4], ignore_index=True)
del part1, part2, part3, part4
gc.collect()

# Conversione Label in valori numerici
for col in df.columns:
    if col != 'Label':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Mappatura dei tipi di attacco
attack_map = {
    'BENIGN': 'BENIGN',
    'DDoS': 'DDoS',
    'DoS Hulk': 'DoS',
    'DoS GoldenEye': 'DoS',
    'DoS slowloris': 'DoS',
    'DoS Slowhttptest': 'DoS',
    'PortScan': 'PortScan',
    'FTP-Patator': 'Brute Force',
    'SSH-Patator': 'Brute Force',
    'Bot': 'Bot',
    'Web Attack - Brute Force': 'Web Attack',
    'Web Attack - XSS': 'Web Attack',
    'Web Attack - Sql Injection': 'Web Attack',
    'Infiltration': 'Infiltration',
    'Heartbleed': 'Heartbleed'
}

# Normalizzazione e pulizia etichette
df['Label'] = df['Label'].astype(str).str.strip()
df['Label'] = df['Label'].str.replace('–', '-', regex=False)
df['Label'] = df['Label'].str.replace(' +', ' ', regex=True)

# Mappatura
df['Attack Type'] = df['Label'].map(lambda x: attack_map.get(x, 'Unknown'))
print("Distribuzione etichette dopo mapping:")
print(df['Attack Type'].value_counts())

# Rimuove classi rare
rare_classes = ['Heartbleed', 'Infiltration']
df = df[~df['Attack Type'].isin(rare_classes)]

# Filtra solo etichette valide
df = df[df['Attack Type'] != 'Unknown']

# Codifica le classi
le = LabelEncoder()
df['Attack Number'] = le.fit_transform(df['Attack Type'])

# Prepara X e y
X = df.drop(['Label', 'Attack Type', 'Attack Number'], axis=1, errors='ignore')
y = df['Attack Number']

# Bilanciamento con RandomUnderSampler
print("Distribuzione originale:", Counter(y))
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)
print("Distribuzione dopo bilanciamento:", Counter(y_resampled))

# Rimozione di colonne non imputabili
non_feature_cols = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp']
# Use X from the previous steps (after label encoding and drop)
X = X.drop(columns=[col for col in non_feature_cols if col in X.columns], errors='ignore')
#Elimino spaziature nomi delle features
X.columns = X.columns.str.strip()

# Codifica di feature categoriche
for col in X.select_dtypes(include=['object']).columns:
    enc = LabelEncoder()
    X[col] = enc.fit_transform(X[col].astype(str))

# Rimpiazzo inf with NaN
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Imputazione valori assenti
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Feature scaling
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Feature selection
selector = SelectKBest(score_func=chi2, k=20)

X_selected = selector.fit_transform(X_scaled, y)
selected_features = X.columns[selector.get_support()]
X = pd.DataFrame(X_selected, columns=selected_features)


# Split (originali)
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Training 1: modello su dati originali
sk_rf_full = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
sk_rf_full.fit(X_train_full, y_train_full)

# Training 2: modello su dati bilanciati
rus = RandomUnderSampler(random_state=42)
X_train_bal, y_train_bal = rus.fit_resample(X_train_full, y_train_full)
X_test_bal, y_test_bal = rus.fit_resample(X_test_full, y_test_full)

sk_rf_bal = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
sk_rf_bal.fit(X_train_bal, y_train_bal)

# Valutazione modelli
def evaluate_model(X_set, y_true, model, name):
    y_pred = model.predict(X_set)
    print(f"\n--- {name} ---")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred, average='weighted'))
    print("Precision:", precision_score(y_true, y_pred, average='weighted'))
    print("Recall:", recall_score(y_true, y_pred, average='weighted'))
    print(classification_report(y_true, y_pred))

evaluate_model(X_test_full, y_test_full, sk_rf_full, "Test Set (Sbilanciato)")
evaluate_model(X_test_bal, y_test_bal, sk_rf_bal, "Test Set (Bilanciato)")

# Libera la memoria
del X, y
gc.collect()

print("Classi viste dal label encoder:", le.classes_)
print("Numero classi:", len(le.classes_))

# Spiegabilità SHAP
def shap_summary(model, data, title="SHAP Summary"):
    background = data.sample(n=100, random_state=42) if len(data) > 100 else data
    explainer = shap.Explainer(model, background)
    shap_values = explainer(background, check_additivity=False)

    plt.figure(figsize=(14, 8))
    shap.summary_plot(shap_values, background, feature_names=background.columns, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()

shap_summary(sk_rf_full, X_test_full, "SHAP - Modello su dati sbilanciati")
shap_summary(sk_rf_bal, X_test_bal, "SHAP - Modello su dati bilanciati")

# Salvataggio del modello sklearn e del codificatore con joblib
try:
    with open('rf_model_imbalanced.joblib', 'wb') as f:
        joblib.dump(sk_rf_full, f)
    with open('rf_model_balanced.joblib', 'wb') as f:
        joblib.dump(sk_rf_bal, f)
    with open('label_encoder.joblib', 'wb') as f:
        joblib.dump(le, f)
    with open('scaler.joblib', 'wb') as f:
        joblib.dump(scaler, f)
    with open('imputer.joblib', 'wb') as f:
        joblib.dump(imputer, f)
    with open('selected_features.joblib', 'wb') as f:
        joblib.dump(selected_features.tolist(), f)
    print("Modello, codificatore e oggetti di preprocessing salvati con joblib.")
except Exception as e:
    print("Errore nel salvataggio dei file:", e)
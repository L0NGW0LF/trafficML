import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate, GridSearchCV, learning_curve, cross_val_predict
from imblearn.over_sampling  import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import seaborn as sns
import joblib
import shap
import gc

def loader(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    df = df[df['Flow ID'] != 'Flow ID']
    return df

# Primo blocco
df1 = loader('../data/raw/training_dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
df2 = loader('../data/raw/training_dataset/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv')
part1 = pd.concat([df1, df2], ignore_index=True)
del df1, df2
gc.collect()

# Secondo blocco
df3 = loader('../data/raw/training_dataset/Friday-WorkingHours-Morning.pcap_ISCX.csv')
df4 = loader('../data/raw/training_dataset/Monday-WorkingHours.pcap_ISCX.csv')
part2 = pd.concat([df3, df4], ignore_index=True)
del df3, df4
gc.collect()

# Terzo blocco
df5 = loader('../data/raw/training_dataset/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv')
df6 = pd.read_csv('../data/raw/training_dataset/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv', low_memory=False, encoding='latin-1')
df6.columns = df6.columns.str.strip()
df6 = df6[df6['Flow ID'] != 'Flow ID']
part3 = pd.concat([df5, df6], ignore_index=True)
del df5, df6
gc.collect()

# Quarto blocco
df7 = loader('../data/raw/training_dataset/Tuesday-WorkingHours.pcap_ISCX.csv')
df8 = loader('../data/raw/training_dataset/Wednesday-workingHours.pcap_ISCX.csv')
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


# Distribuzione delle classi (countplot)
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Attack Type', order=df['Attack Type'].value_counts().index)
plt.xticks(rotation=45)
plt.title("Distribuzione degli Attacchi")
plt.tight_layout()
plt.show()
plt.close()

# Codifica le classi
le = LabelEncoder()
df['Attack Number'] = le.fit_transform(df['Attack Type'])

# Prepara X e y
X = df.drop(['Label', 'Attack Type', 'Attack Number'], axis=1, errors='ignore')
y = df['Attack Number']

del df
gc.collect()

# Rimozione di colonne non imputabili
non_feature_cols = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp']
X = X.drop(columns=[col for col in non_feature_cols if col in X.columns], errors='ignore')
X.columns = X.columns.str.strip()

# Codifica di feature categoriche (se presenti dopo la pulizia iniziale)
for col in X.select_dtypes(include=['object']).columns:
    enc = LabelEncoder()
    X[col] = enc.fit_transform(X[col].astype(str))

# Gestione di valori infiniti e NaN
X.replace([np.inf, -np.inf], np.nan, inplace=True)
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Scalatura delle feature
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Selezione delle feature con chi-quadro
selector = SelectKBest(score_func=chi2, k=20)
X_selected = selector.fit_transform(X_scaled, y)
selected_features = X.columns[selector.get_support()]

# Dataset finale con le sole feature selezionate
X = pd.DataFrame(X_selected, columns=selected_features)

print("Shape del dataset dopo la selezione delle feature:", X.shape)
print("Feature selezionate:", list(X.columns))

plt.figure(figsize=(12, 10))
sns.heatmap(X.corr(), annot=False, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Matrice di Correlazione tra le Feature Selezionate")
plt.tight_layout()
plt.show()
plt.close()

# Impostazioni per la cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Modello per il dataset sbilanciato
sk_rf_full = RandomForestClassifier(n_estimators=50, max_depth=20, random_state=42)

# Esecuzione della cross-validation
print("Cross-validation - Modello Sbilanciato")
score_sbil = cross_val_score(sk_rf_full, X, y, cv=cv, scoring='f1_weighted', n_jobs=-1)

print(f"F1 Weighted - Media: {score_sbil.mean():.4f}")
print(f"F1 Weighted - Deviazione standard: {score_sbil.std():.4f}")

# Sottocampionamento (Undersampling) delle classi con più di 100k istanze
sampling_strategy_under = {
    le.transform(['BENIGN'])[0]: 100000,
    le.transform(['DoS'])[0]: 100000,
    le.transform(['PortScan'])[0]: 100000,
    le.transform(['DDoS'])[0]: 100000
}
rus = RandomUnderSampler(sampling_strategy=sampling_strategy_under, random_state=42)
X_resampled_under, y_resampled_under = rus.fit_resample(X, y)
print("Distribuzione originale:", Counter(y))
print("Distribuzione dopo l'undersampling:", Counter(y_resampled_under))

# Sovracampionamento (Oversampling) delle classi minoritarie
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_resampled_under, y_resampled_under)
print("\nDistribuzione finale bilanciata:", Counter(y_resampled))

# Definizione il modello base senza iperparametri fissi
rf = RandomForestClassifier(random_state=42)

# Definizione della griglia di iperparametri
param_grid = {
    'n_estimators': [50, 100, 150],  # Prova con 50, 100 e 150 alberi
    'max_depth': [20, 30, None],     # Prova con profondità 20, 30 e illimitata (None)
    'min_samples_leaf': [1, 2, 4]      # Prova con un numero minimo di campioni per foglia
}

#    Imposta GridSearchCV
#    Usiamo la stessa cross-validation stratificata (cv) definita prima.
grid_search = GridSearchCV(estimator=rf,
                           param_grid=param_grid,
                           cv=cv,
                           scoring='f1_weighted',
                           n_jobs=-1,  # Usa tutti i core disponibili
                           verbose=2) # Mostra l'avanzamento

# Avvia la ricerca degli iperparametri ottimali
print("Avvio di GridSearchCV per trovare i migliori iperparametri...")
grid_search.fit(X_resampled, y_resampled)

# Stampa i risultati migliori
print("\nMigliori iperparametri trovati:")
print(grid_search.best_params_)

# Stampa il miglior punteggio F1 pesato ottenuto
print("\nMiglior F1-score (pesato) ottenuto in cross-validation:")
print(f"{grid_search.best_score_:.4f}")

#Salvataggio del modello
sk_rf_bal = grid_search.best_estimator_

# Calcolo F1 score per il modello bilanciato
score_bal = cross_val_score(sk_rf_bal, X_resampled, y_resampled, cv=cv, scoring='f1_weighted', n_jobs=-1)

# Barplot comparativo dei risultati F1 Score
plt.figure(figsize=(8, 5))
sns.barplot(data=pd.DataFrame({
    'Score': np.concatenate([score_sbil, score_bal]),
    'Tipo': ['Sbilanciato'] * len(score_sbil) + ['Bilanciato'] * len(score_bal)
}), x='Tipo', y='Score', errorbar='sd')
plt.title("Confronto F1 Score tra Modello Sbilanciato e Bilanciato (CV)")
plt.ylabel("F1 Score (Weighted)")
plt.tight_layout()
plt.show()
plt.close()

# Valutazione completa - Modello Sbilanciato
print("\nValutazione completa - Modello Sbilanciato (CV)")
scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
cv_results_sbil = cross_validate(sk_rf_full, X, y, cv=cv, scoring=scoring, n_jobs=-1)
for metric in scoring:
    mean = cv_results_sbil['test_' + metric].mean()
    std = cv_results_sbil['test_' + metric].std()
    print(f"{metric}: {mean:.4f} ± {std:.4f}")

# Valutazione completa - Modello Bilanciato
print("\nValutazione completa - Modello Bilanciato (CV)")
cv_results_bal = cross_validate(sk_rf_bal, X_resampled, y_resampled, cv=cv, scoring=scoring, n_jobs=-1)
for metric in scoring:
    mean = cv_results_bal['test_' + metric].mean()
    std = cv_results_bal['test_' + metric].std()
    print(f"{metric}: {mean:.4f} ± {std:.4f}")

def print_cv_metrics_and_confusion_matrix(model, X, y, cv, class_names, title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)
    y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
    # Classification report (media per classe)
    report = classification_report(y, y_pred, target_names=class_names, digits=4, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    print(df_report[['precision', 'recall', 'f1-score', 'support']])
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {title}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    plt.close()

# Valutazione dettagliata modello sbilanciato
print_cv_metrics_and_confusion_matrix(sk_rf_full, X, y, cv, le.classes_, "Modello Sbilanciato (CV)")

# Valutazione dettagliata modello bilanciato
print_cv_metrics_and_confusion_matrix(sk_rf_bal, X_resampled, y_resampled, cv, le.classes_, "Modello Bilanciato (CV)")

# Funzione per plottare la curva di apprendimento
def plot_learning_curve(estimator, title, X, y, cv):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y,
        cv=cv, scoring='f1_weighted',
        train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.title(title)
    plt.xlabel("Dimensione del Training Set")
    plt.ylabel("F1 Score (Weighted)")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()

# Generazione curve per entrambi i modelli
plot_learning_curve(sk_rf_full, "Curva di Apprendimento (Modello Sbilanciato)", X, y, cv)
plot_learning_curve(sk_rf_bal, "Curva di Apprendimento (Modello Bilanciato)", X_resampled, y_resampled, cv)

# Addestramento dei modelli sui rispettivi dataset
sk_rf_full.fit(X, y)
sk_rf_bal.fit(X_resampled, y_resampled)

# Importanza delle feature - Sbilanciato
importances_full = sk_rf_full.feature_importances_
feature_importance_df_full = pd.DataFrame({'feature': selected_features, 'importance': importances_full}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='importance', y='feature', data=feature_importance_df_full)
plt.title("Importanza delle Feature - Modello Sbilanciato")
plt.tight_layout()
plt.show()
plt.close()

# Importanza delle feature - Bilanciato
importances_bal = sk_rf_bal.feature_importances_
feature_importance_df_bal = pd.DataFrame({'feature': selected_features, 'importance': importances_bal}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='importance', y='feature', data=feature_importance_df_bal)
plt.title("Importanza delle Feature - Modello Bilanciato")
plt.tight_layout()
plt.show()
plt.close()

print("Classi viste dal label encoder:", le.classes_)
print("Numero classi:", len(le.classes_))

# Spiegabilità SHAP
def shap_summary(model, data, title_prefix="SHAP Summary"):
    background = data.sample(n=100, random_state=42) if len(data) > 100 else data

    # Usa TreeExplainer con output in probabilità per compatibilità multiclass
    explainer = shap.Explainer(model, background, model_output="probability", algorithm="tree")
    shap_values = explainer(background)

    # Calcola shap_values per la classe più rappresentata
    if isinstance(shap_values.values, list) or shap_values.values.ndim == 3:
        # Se multi-classe, seleziona la classe 0 come esempio (modifica se necessario)
        class_index = 0
        values = shap_values[..., class_index]
    else:
        values = shap_values

    # Beeswarm plot
    plt.figure(figsize=(14, 8))
    shap.summary_plot(values, background, plot_type="dot", feature_names=background.columns, show=False)
    plt.title(f"{title_prefix} - Beeswarm")
    plt.tight_layout()
    plt.show()

    # Violin plot
    plt.figure(figsize=(14, 8))
    shap.summary_plot(values, background, plot_type="violin", feature_names=background.columns, show=False)
    plt.title(f"{title_prefix} - Violin")
    plt.tight_layout()
    plt.show()

    # Bar plot
    plt.figure(figsize=(14, 6))
    shap.summary_plot(values, background, plot_type="bar", feature_names=background.columns, show=False)
    plt.title(f"{title_prefix} - Bar (Feature Importance)")
    plt.tight_layout()
    plt.show()


shap_summary(sk_rf_full, X, "SHAP - Modello su dati sbilanciati")
shap_summary(sk_rf_bal, X_resampled, "SHAP - Modello su dati bilanciati")

# Salvataggio del modello sklearn e del codificatore con joblib
try:
    with open('../out/model/rf_model_imbalanced.joblib', 'wb') as f:
        joblib.dump(sk_rf_full, f)
    with open('../out/model/rf_model_balanced.joblib', 'wb') as f:
        joblib.dump(sk_rf_bal, f)
    with open('../out/model/label_encoder.joblib', 'wb') as f:
        joblib.dump(le, f)
    with open('../out/model/scaler.joblib', 'wb') as f:
        joblib.dump(scaler, f)
    with open('../out/model/imputer.joblib', 'wb') as f:
        joblib.dump(imputer, f)
    with open('../out/model/selected_features.joblib', 'wb') as f:
        joblib.dump(selected_features.tolist(), f)
    print("Modello, codificatore e oggetti di preprocessing salvati con joblib.")
except Exception as e:
    print("Errore nel salvataggio dei file:", e)
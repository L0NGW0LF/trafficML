
# TrafficML: Intrusion Detection using Machine Learning

TrafficML is a machine learning pipeline designed to detect network intrusions using the CIC IDS 2017 dataset. This pipeline performs data preprocessing, feature selection, model training, and prediction.

## 📊 Dataset

The dataset used is the [CIC IDS 2017](https://www.unb.ca/cic/datasets/ids-2017.html), provided by the Canadian Institute for Cybersecurity. It includes various types of attacks like DoS, DDoS, PortScan, BruteForce, Web Attacks, and more, with realistic network traffic captured over several days.

## ⚙️ Technologies

- Python 3
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- imbalanced-learn
- joblib
- Google Colab / Jupyter Notebook

## 🚦 Pipeline Overview

1. **Data Loading** – Multiple CSV files are loaded and concatenated.
2. **Data Cleaning** – Headers are stripped, repeated rows removed, and invalid values handled.
3. **Feature Engineering** – Categorical encoding, missing value imputation, scaling, and feature selection.
4. **Model Training** – RandomForestClassifier with undersampling and GridSearch (or cuML for GPU training).
5. **Evaluation** – Metrics like Accuracy, F1, Precision, Recall.
6. **Export** – Save model, label encoder, imputer, scaler, and selected features.

## 🧪 How to Run

1. Clone the repository and navigate into the project folder.
2. Make sure all required dependencies are installed:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the notebook or script:
    ```bash
    jupyter notebook TrafficML_pipeline.ipynb
    ```
    or, for the inference script:
    ```bash
    python trafficml_pipeline.py
    ```

## 📁 File Structure

- `TrafficML_pipeline.ipynb` – Main notebook for training.
- `main.py` – Python script to classify new network flows.
- `data/` – Contains raw CSVs and processed model artifacts.
- `out/` - Contains attacks logs and trained model with encoded labels, etc...
- `docs/` - Contains documentation regarding the choices through the process of training to the deployment

## ✅ License

This project is open source under the MIT License.

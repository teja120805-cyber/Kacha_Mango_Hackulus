
1)Single-Lead AF Detection with train_precision_ecg.py

File:train_precision_ecg.py


Overview

This project trains a single-lead ECG model to detect Atrial Fibrillation (AF) using the PhysioNet 2017 dataset.

The model is implemented in PyTorch.

It uses weighted loss and oversampling to handle the imbalanced dataset.

Outputs:

Trained model (model.pt)

Metrics CSV (metrics.csv)

Prerequisites

Python 3.10+ installed.

Virtual environment recommended to isolate dependencies.

Required Python packages:

pip install torch torchvision torchaudio
pip install pandas scikit-learn numpy

Setup Instructions
1️⃣ Create and activate virtual environment (optional but recommended)
D:\> python -m venv venv
D:\> venv\Scripts\activate


You should see (venv) in the prompt.

2️⃣ Place CSV files

Ensure your CSV files are in the correct location:

ECG data CSV: D:\Physionet2017ECGData.csv

ECG labels CSV: D:\Physionet2017ECGLabels.csv

The CSV format:

Data CSV: 5655 ECGs, 9000 samples each, columns named 0..MHD

Label CSV: single column of labels A (AF) or N (Not AF)

3️⃣ Place training script

Make sure train_precision_ecg.py is in D:\ (or update paths accordingly).

4️⃣ Run training

Open Command Prompt, activate the virtual environment, and run:

(venv) D:\> python train_precision_ecg.py --data_csv "D:\Physionet2017ECGData.csv" --label_csv "D:\Physionet2017ECGLabels.csv" --out "D:\ECG_out"


--data_csv: Path to ECG data

--label_csv: Path to labels

--out: Output folder for trained model and metrics

5️⃣ Output

After training, the following will be created:

D:\ECG_out\model.pt → Trained PyTorch model

D:\ECG_out\metrics.csv → Test set results including:

Precision, Recall, F1

Confusion matrix

Sensitivity and specificity

6️⃣ Optional Notes

Adjusting batch size: Add --batch <number> if needed. Default: 32

Adjusting epochs: Add --epochs <number> if more training is desired. Default: 25

Device: Script automatically uses GPU if available; fallback to CPU otherwise

7️⃣ Troubleshooting

Missing packages: Use pip install <package> for any missing modules

Virtual environment not activating: Ensure the path is correct: D:\venv\Scripts\activate

Paths contain spaces: Keep paths in quotes, e.g., "D:\Physionet2017ECGData.csv"

8️⃣ References

PhysioNet 2017 Challenge Dataset: https://www.kaggle.com/datasets/luigisaetta/physionet2017ecg


____________________________________________________________________________________________________________________________________________________________________


2)Sepsis Prediction using Random Forest

File:sepsisdata_code.py

This project implements a Random Forest classifier to predict the risk of sepsis based on patient vital signs. The workflow includes data preprocessing, model training, evaluation, and saving the trained model for future use.

Table of Contents

Overview

Dataset

Features

Installation

Usage

Model Evaluation

Saved Model

License

Overview

The script sepsis_prediction.py performs the following steps:

Loads the sepsis dataset from a CSV file.

Preprocesses features by imputing missing values and scaling them.

Splits the data into training (first 1500 samples) and validation (next 500 samples) sets.

Trains a Random Forest classifier with class balancing.

Evaluates the model on the validation set using metrics like accuracy, precision, recall, and ROC AUC.

Saves the trained model and preprocessing pipeline to a pickle file (sepsis_rf_model.pkl) for later use.

Dataset

File: cloned_sepsis_dataset.csv

Location: C:\Users\Shrey\Downloads\cloned_sepsis_dataset.csv (update path as needed)

Target column: sepsis_label

Features used:

heart_rate

respiratory_rate

body_temperature

oxygen_saturation

Note: Column names are standardized to lowercase without spaces during preprocessing.

Features

The following patient vitals are used as input features:

Feature	Description
heart_rate	Beats per minute (BPM)
respiratory_rate	Breaths per minute
body_temperature	Temperature in Celsius
oxygen_saturation	Blood oxygen level (%)

Target: sepsis_label (binary: 0 = no sepsis, 1 = sepsis)

Installation

Clone this repository.

Ensure Python 3.8+ is installed.

Install required packages:

pip install pandas scikit-learn

Usage

Update the dataset path in the script if needed:

file_path = r"C:\path\to\cloned_sepsis_dataset.csv"


Run the script:

python sepsis_prediction.py


The script outputs:

Accuracy, Precision, Recall, ROC AUC

Full classification report

Saves trained model as sepsis_rf_model.pkl

Model Evaluation

Example output on validation set (500 samples):

Accuracy : 0.850
Precision: 0.812
Recall   : 0.780
ROC AUC  : 0.910

Classification Report:
              precision    recall  f1-score   support
           0       0.87      0.88      0.87       300
           1       0.81      0.78      0.80       200


Metrics may vary based on the dataset used.

Saved Model

The trained model and preprocessing pipeline are saved in sepsis_rf_model.pkl:

Contains:

model: Random Forest classifier

preprocessor: Imputer + Scaler pipeline

features: List of input features

To load the model for prediction:

import pickle

with open("sepsis_rf_model.pkl", "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
preprocessor = saved["preprocessor"]
features = saved["features"]

# Example prediction
X_new_prep = preprocessor.transform(X_new)  # X_new = new data in same format
y_pred = model.predict(X_new_prep)

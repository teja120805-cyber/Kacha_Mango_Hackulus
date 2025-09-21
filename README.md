# Kacha_Mango_Hackulus
README: Single-Lead AF Detection with train_precision_ecg.py
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

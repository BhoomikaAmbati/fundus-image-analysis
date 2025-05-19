# Fundus Image Analysis

This repository contains all necessary scripts, data, and results for training, validating, and analyzing a deep learning model using augmented and original datasets. Below is a structured breakdown of the project directory and its components.

## 📂 Directory Structure

```
/ProjectRoot
│── Data/              # Contains augmented dataset used for training
│── new_data/          # Original, non-augmented dataset
│── logs/              # (Ignore) Stores training and system logs
│── model/             # Stores the trained model (`model.h5`)
│── model_build/       # Contains model definition scripts
│── pca/               # Scripts for PCA analysis on model outputs
│── results/           # Stores various results from validation and PCA
│   ├── validation_results/  # Extracted validation results
│   ├── pca_results/        # Principal component analysis results
│── train/             # Training scripts and metrics computation
│   ├── train.py       # Script for training the model using augmented data
│   ├── metrics.py     # Computes and logs training metrics
│── validation/        # Evaluation scripts for model validation
│   ├── eval.py        # Extracts validation results and stores them
│── unet_model_summary.pdf  # Model architecture summary
│── .gitignore         # Git ignore file
│── readme.md          # Project documentation
```

## Key Components

### 🔹 **Data Handling**

- `Data/` → Contains the augmented dataset used for training.
- `new_data/` → Contains the original, unaltered dataset.

### 🔹 **Model Training**

- `train/train.py` → Loads the augmented dataset, trains the model, and saves it as `model/model.h5`.
- `train/metrics.py` → Computes training metrics and logs progress.

### 🔹 **Model Evaluation**

- `validation/eval.py` → Evaluates the trained model on validation data and stores results in `results/validation_results/`.

### 🔹 **Principal Component Analysis (PCA)**

- `pca/` → Scripts for PCA computation on hidden layer outputs.
- Outputs stored in `results/pca_results/`.

### 🔹 **Model Architecture**

- `unet_model_summary.pdf` → Provides a summary of the U-Net model architecture.

## 🚀 Getting Started

1. Clone the repository:
   ```sh
   git clone <repository-url>
   cd ProjectRoot
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Train the model:
   ```sh
   python train/train.py
   ```
4. Validate the model:
   ```sh
   python validation/eval.py
   ```
5. Perform PCA analysis:
   ```sh
   python pca/pca_analysis.py
   ```

## 📜 Notes

- The training process generates a trained model (`model/model.h5`).
- Validation results are stored under `results/validation_results/`.
- PCA analysis is performed on hidden layer outputs and results are stored in `results/pca_results/`.

---

This repository provides a complete pipeline for training, validating, and analyzing a deep learning model with both augmented and original datasets. 🎯


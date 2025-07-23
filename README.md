# Credit Risk Modeling: BlackSwan vs. Random Forest vs. Gradient Boosting

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](#)
[![License](https://img.shields.io/badge/license-MIT-green)](#)

## 📖 Overview

This project compares the performance of three classifiers on the [Kaggle Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset/data):

1. **BlackSwan** (proprietary algorithm)
2. **Random Forest**
3. **Gradient Boosting**

We evaluate them on a held‑out backtest set using AUC‑based metrics, calibration, and threshold‑dependent metrics.

## 🗃️ Dataset

- **Source**: Kaggle | `laotse/credit-risk-dataset`  
- **Description**: Contains customer demographics, loan application details, and repayment status.
- **Preprocessing**:
  - Train/validation/test split: 20% / 0% / 80%.

Directory structure for raw data:

```
data-room/
├── training.csv
├── backtest.csv
└── backtest_with_predictions.csv
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- `pip` or `conda`

### Installation

```bash
git clone https://github.com/AlgorithmeAi/Credit-Risk-Dataset-BlackSwan-vs-Random-Forest-vs-Algorithme.git
cd Credit-Risk-Dataset-BlackSwan-vs-Random-Forest-vs-Algorithme
```

`requirements.txt` includes:
```
numpy
pandas
scikit-learn
matplotlib  # (hypothetical SDK for BlackSwan)
```

## 📂 Project Structure

```
.
├── data-room/              # Raw and processed data
├── results/
│   └── benchmark.pdf       # Summary of AUC Based Metrics for algorithme90
├── tools/
│   ├── automl.py           # Performs RF/GB over Train Test Target
│   ├── plot_model.py       # Plots the AUC Metrics from Augmented Backtest
│   └── prep-maker.py       # Makes a training and a backtest out of a single .csv file
├── requirements.txt
├── algorithme15.py     # 15 Cores BlackSwan Model
├── algorithme90.py     # 90 Cores BlackSwan Model
├── demo-script.py      # Script for demo the Audit Capabilities of BlackSwan
└── README.md

```

## 🛠️ Usage

1. **Demo Algorithme Audit Features**

   ```bash
   python demo-script.py 
   ```

2. **Extrapolate on Backtest**

   ```bash
   python algorithme15.py extrapolate backtest.csv
   ```
   
   ```bash
   python algorithme90.py extrapolate backtest.csv
   ```

## 📊 Results

All metrics computed on the **backtest set**:

| Metric                  | BlackSwan | Random Forest | Gradient Boosting |
|-------------------------|----------:|--------------:|------------------:|
| **ROC AUC**             |    0.901  |        0.888  |           0.919  |
| **Precision‑Recall AUC**|    0.821  |        0.543  |           0.822  |
| **Recall AUC**          |    0.551  |        0.308  |           0.632  |
| **F1‑score AUC**        |    0.570  |        0.273  |           0.656  |
| **Youden’s J (max)**    |    0.60   |        0.56   |           0.62   |
| **Calibration (Brier)** |    0.15   |        0.18   |           0.14   |

> **Key takeaways**  
> - Gradient Boosting achieves the highest ROC AUC (0.919) and F1‑score AUC (0.656).  
> - BlackSwan outperforms Random Forest on most AUC‑based metrics, but is slightly behind Gradient Boosting on ROC AUC.  
> - Calibration curves show BlackSwan and Gradient Boosting are better calibrated than Random Forest.

## 🧪 Experimental Setup

- **Hardware**: APPLE M4 MAX, 128 GB RAM

## 🤝 Contributing

Contributions welcome! Please open issues or PRs to:

- Add new models (e.g., LightGBM, CatBoost).
- Extend evaluation (e.g., fairness metrics, cost-sensitive analysis).
- Create an Ai Agent for Credit Risk.

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

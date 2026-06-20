# Anomaly Detection in Network Traffic Using Supervised Machine Learning

This project detects malicious network traffic (intrusions/anomalies) using the **UNSW-NB15** dataset. It compares three classical machine learning models — Random Forest, Decision Tree, and SVM — against a custom **PyTorch Multi-Layer Perceptron (MLP)**, and includes multi-class attack categorization, hyperparameter tuning, model explainability, and a deployment-ready inference pipeline.

## What's in this notebook

The notebook (`Anomaly_Detection_Final-using_MLP-updated--.ipynb`) runs end-to-end and is organized into these sections:

1. **Data loading** — Reads the UNSW-NB15 training/testing CSVs.
2. **EDA** — Class distribution, missing values, protocol distribution, correlation heatmap, feature distributions, byte/TTL analysis (plots saved as `.png`).
3. **Preprocessing** — Label encoding of categorical features, median imputation, `StandardScaler` feature scaling.
4. **Binary classification (Normal vs. Attack)** — Random Forest, Decision Tree, and SVM, each with confusion matrices, ROC curves, and 5-fold cross-validation.
5. **Extended work**:
   - Multi-class classification (predicting the specific attack category, e.g. DoS, Exploits, Reconnaissance, Generic, Fuzzers)
   - Hyperparameter tuning via `GridSearchCV` (Random Forest, SVM)
   - Explainability via permutation importance and SHAP (where applicable)
   - A deployment-ready `sklearn` `Pipeline` saved with `joblib`, plus a `predict_traffic_status()` helper function and latency/throughput benchmarking
6. **Neural network (MLP)** — A custom PyTorch model (`NetworkDetector`) with batch norm, dropout, learning-rate scheduling, and early stopping, including a small hyperparameter search.
7. **Final comparison** — Accuracy, AUC-ROC, and ROC curves across all four models (RF, DT, SVM, MLP).

## Dataset

This project uses the **UNSW-NB15** dataset, created by the Australian Centre for Cyber Security (ACCS).

- Download the training and testing CSVs from: https://research.unsw.edu.au/projects/unsw-nb15-dataset
- You need:
  - `UNSW_NB15_training-set.csv`
  - `UNSW_NB15_testing-set.csv`
- Place both files in the **same folder as the notebook**. The notebook auto-detects the files if they're in the working directory; otherwise edit the file paths in the "Load Dataset" cell.
- The dataset is **not included** in this repository — you must download it separately (it's a large, third-party academic dataset).

## Requirements

- Python 3.9+
- Jupyter Notebook or JupyterLab

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch torchvision torchaudio joblib shap
```

> `torch` is installed automatically from within the notebook (Section 15) if not already present. `shap` is optional — the notebook falls back to permutation importance if it isn't installed.

## How to run

1. Clone this repository and `cd` into it.
2. Install the dependencies above (a virtual environment is recommended).
3. Download the UNSW-NB15 CSVs (see **Dataset** above) and place them in the project root.
4. Launch Jupyter and open the notebook:
   ```bash
   jupyter notebook "Anomaly_Detection_Final-using_MLP-updated--.ipynb"
   ```
5. Run all cells from top to bottom (**Kernel → Restart & Run All**). The notebook is sequential — later sections (multi-class classification, tuning, the MLP) depend on variables created earlier (e.g. `X_train`, `rf`, `svm`), so cells should not be run out of order.

Running the full notebook will generate:
- A set of `plot_*.png` files (EDA charts, confusion matrices, ROC curves, training curves) saved to the working directory.
- A saved model package: `unsw_nb15_binary_deployment_pipeline.joblib` (preprocessing + tuned SVM pipeline for binary classification).

## Reusing the trained model

After running the notebook, the binary classifier can be reloaded and used standalone:

```python
import joblib
import pandas as pd

package = joblib.load("unsw_nb15_binary_deployment_pipeline.joblib")
pipeline = package["pipeline"]
feature_columns = package["feature_columns"]

new_data = pd.read_csv("your_traffic_records.csv")
predictions = pipeline.predict(new_data[feature_columns])
probabilities = pipeline.predict_proba(new_data[feature_columns])[:, 1]
```

The input data must contain the same raw feature columns as the original UNSW-NB15 records (the pipeline handles scaling/encoding internally).

## Notes for other users

- Random seeds are fixed (`random_state=42`) throughout for reproducibility.
- The binary models are trained on a **balanced sample** (500 normal + up to 500 records per selected attack category), not the full dataset — this keeps training fast and classes manageable. Adjust the sampling in Section 4 if you want to use the full dataset or different attack categories.
- The MLP (Section 15) trains noticeably slower on CPU; a GPU is not required but will speed things up if `torch` detects one.
- GridSearchCV in Section 14.3 can take several minutes depending on hardware — reduce the parameter grids if you just want a quick run-through.

## License

Specify a license for your repository (e.g. MIT) if you intend others to reuse this code. The UNSW-NB15 dataset has its own usage terms set by UNSW — refer to the dataset's official page for citation and usage requirements.

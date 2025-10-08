# DSA



# Hb Prediction from Lip Images Using SSL + XGBoost

Predict haemoglobin (Hb) levels from lip images using a **Self-Supervised Learning (SSL) backbone** for feature extraction and an **XGBoost regressor** for final prediction. This repository supports **training**, **hyperparameter optimization**, and **inference**.

---

## 📂 Project Structure

```
dsa/                     # Root folder
├── models/              # Saved model weights and XGBoost models
│   ├── final_ssl_backbone.pth
│   ├── final_ssl_proj_head.pth
│   └── final_xgb_model.json
├── src/                 # Source code
│   └── inference.py             # Run inference on new images
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## ⚡ Features

* **SSL Backbone Pretraining**: Pretrain a ResNet-based SSL model on labeled/unlabeled lip images.
* **Fine-Tuning**: Optionally fine-tune the backbone on labeled data.
* **XGBoost Regression**: Train on embeddings to predict Hb levels.
* **Hyperparameter Optimization**: Optuna-based tuning to minimize MAE, with metrics for R² and RMSE.
* **Inference Pipeline**: Load pretrained models and predict Hb for new images.
* **Supports Multiple Image Formats**: `.jpg`, `.jpeg`, `.png`, `.heic`, `.heif`.

---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/hb-prediction.git
cd hb-prediction
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🧪 Inference

Run inference on new images using the pretrained models:

```bash
python src/inference.py \
    --image_dir path/to/test_images \
    --model_dir models \
    --batch_size 8 \
    --output_csv predictions.csv
```

**Outputs**:

* CSV with predicted Hb levels for each image.

---

## 📝 Example Usage

```python
from src.inference import run_inference

predictions_df = run_inference(
    image_dir="data/test_images",
    model_dir="models",
    batch_size=8,
    output_csv="predictions.csv"
)

print(predictions_df.head())
```

---

## 🔧 Dependencies

Key Python packages are listed in `requirements.txt`:

* `torch` / `torchvision`
* `xgboost`
* `optuna`
* `pandas`, `numpy`
* `Pillow`, `pillow-heif`

---

## 🗂 Notes


---



# CodeAlpha_CarPricePrediction

**CodeAlpha Data Science Internship — Task 3: Car Price Prediction with Machine Learning**

## 📖 Overview
This project trains regression models to predict a car's market price from
its specs: brand, model year, horsepower, engine size, fuel efficiency,
odometer reading, fuel type, and transmission.

## ⚠️ About the dataset
The CodeAlpha task PDF's "DOWNLOAD DATASET FROM here" link is a bare
placeholder with no resolvable URL. `generate_dataset.py` builds a
**realistic stand-in** (1,000 listings across 18 brands, 2005–2024 model
years) where price is computed from a genuine underlying pricing model
(brand goodwill × specs − depreciation, plus market noise), so the ML
models have real, learnable relationships rather than pure noise.

**To use a real dataset instead** (e.g. Kaggle's `CarPrice_Assignment.csv`
or any similar listing data): save it as `data/car_data.csv` with columns
`brand, year, fuel_type, transmission, horsepower, engine_size_l,
mileage_kmpl, odometer_km, price` (rename columns to match if needed) and
re-run `car_price_prediction.py` — no code changes needed.

## 📂 Project Structure
```
CodeAlpha_CarPricePrediction/
├── generate_dataset.py       # Builds data/car_data.csv
├── car_price_prediction.py   # Main script: EDA, preprocessing, training, evaluation
├── requirements.txt
├── results.json              # Model metrics, generated on run
├── data/
│   └── car_data.csv
├── plots/
│   ├── price_distribution.png
│   ├── price_by_brand.png
│   ├── correlation_heatmap.png
│   ├── price_relationships.png
│   ├── model_comparison.png
│   ├── feature_importance.png
│   └── actual_vs_predicted_*.png (one per model)
└── models/
    └── best_model.pkl         # Full sklearn pipeline (preprocessing + model)
```

## ⚙️ Setup
```bash
pip install -r requirements.txt
```

## ▶️ Run
```bash
python generate_dataset.py        # only needed if data/car_data.csv doesn't already exist
python car_price_prediction.py
```

## 📊 Results
| Model | R² | RMSE | MAE |
|---|---|---|---|
| Linear Regression | 0.990 | $1,037 | $819 |
| **Ridge Regression** | **0.990** | **$1,029** | **$815** |
| Lasso Regression | 0.990 | $1,035 | $817 |
| Random Forest | 0.894 | $3,414 | $2,723 |
| Gradient Boosting | 0.973 | $1,725 | $1,397 |

**Best model: Ridge Regression** — R² = 0.990 on the held-out test set.
The linear models outperform the tree ensembles here because the
underlying price relationship (brand goodwill × specs, minus
depreciation) is close to linear — a useful reminder that more complex
models aren't always better, and it's worth trying simple baselines first.

## 🧠 Key Concepts Demonstrated
- Feature engineering (deriving `car_age` from `year`)
- Preprocessing pipeline: `StandardScaler` for numeric features,
  `OneHotEncoder` for categoricals, combined via `ColumnTransformer`
- Training and comparing 5 regression algorithms
- Evaluation: R², RMSE, MAE, 5-fold cross-validation
- Feature importance analysis
- Model persistence with the full pipeline (`joblib`), so predictions on
  new raw data don't require re-doing preprocessing by hand

## ✅ CodeAlpha Submission Checklist
- [x] Source code uploaded to a GitHub repo named `CodeAlpha_CarPricePrediction`
- [ ] Post project status update on LinkedIn, tagging **@CodeAlpha**
- [ ] Post a video walkthrough of the project on LinkedIn with the GitHub repo link
- [ ] Submit the completed task via the official Submission Form (shared in the WhatsApp group)

---
*Built for the CodeAlpha Data Science Internship program.*

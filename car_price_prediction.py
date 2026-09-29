"""
CodeAlpha Data Science Internship — Task 3
Car Price Prediction with Machine Learning

This script:
  1. Loads and explores car listing data (brand, specs, price)
  2. Performs preprocessing: feature engineering + encoding categoricals
  3. Trains several regression models to predict price
  4. Evaluates and compares performance (R2, RMSE, MAE)
  5. Saves the best model and feature-importance / diagnostic plots

Run:
    python car_price_prediction.py
"""

import os
import json
import warnings

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "car_data.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")
RANDOM_STATE = 42
TARGET = "price"
NUMERIC_FEATURES = ["year", "horsepower", "engine_size_l", "mileage_kmpl", "odometer_km", "car_age"]
CATEGORICAL_FEATURES = ["brand", "fuel_type", "transmission"]


def load_and_engineer():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna().drop_duplicates()
    df["car_age"] = 2025 - df["year"]
    return df


def explore(df):
    print("=" * 60)
    print("DATA OVERVIEW")
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print(df.describe(include="all").T[["count", "mean", "std", "min", "max"]] if False else df.describe())
    print("\nMissing values:\n", df.isnull().sum())

    plt.figure(figsize=(7, 5))
    sns.histplot(df[TARGET], kde=True, color="#3182bd")
    plt.title("Distribution of Car Prices")
    plt.xlabel("Price ($)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "price_distribution.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    order = df.groupby("brand")[TARGET].mean().sort_values(ascending=False).index
    sns.boxplot(data=df, x="brand", y=TARGET, order=order, hue="brand", legend=False, palette="viridis")
    plt.xticks(rotation=60, ha="right")
    plt.title("Price Distribution by Brand")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "price_by_brand.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 5))
    corr = df[NUMERIC_FEATURES + [TARGET]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col in zip(axes, ["horsepower", "car_age", "odometer_km"]):
        sns.scatterplot(data=df, x=col, y=TARGET, hue="fuel_type", alpha=0.5, ax=ax, legend=(col == "odometer_km"))
        ax.set_title(f"Price vs. {col}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "price_relationships.png"), dpi=150)
    plt.close()


def build_preprocessor():
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def train_and_evaluate(df):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, random_state=RANDOM_STATE),
    }

    results = {}
    best_name, best_pipeline, best_r2 = None, None, -np.inf

    print("\n" + "=" * 60)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 60)

    for name, model in models.items():
        pipeline = Pipeline([("prep", build_preprocessor()), ("model", model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="r2")

        results[name] = {
            "R2": round(float(r2), 4),
            "RMSE": round(float(rmse), 2),
            "MAE": round(float(mae), 2),
            "CV_R2_mean": round(float(cv_scores.mean()), 4),
        }
        print(f"\n{name}")
        print(f"  R2: {r2:.4f}  |  RMSE: ${rmse:,.2f}  |  MAE: ${mae:,.2f}  |  CV R2: {cv_scores.mean():.4f}")

        if r2 > best_r2:
            best_name, best_pipeline, best_r2 = name, pipeline, r2

        plt.figure(figsize=(5.5, 5.5))
        plt.scatter(y_test, y_pred, alpha=0.4, color="#2b6cb0")
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        plt.plot(lims, lims, "r--", linewidth=1.5)
        plt.xlabel("Actual Price ($)")
        plt.ylabel("Predicted Price ($)")
        plt.title(f"Actual vs. Predicted — {name}")
        plt.tight_layout()
        safe_name = name.lower().replace(" ", "_")
        plt.savefig(os.path.join(PLOTS_DIR, f"actual_vs_predicted_{safe_name}.png"), dpi=150)
        plt.close()

    # Model comparison chart
    plt.figure(figsize=(7, 4))
    names = list(results.keys())
    r2s = [results[n]["R2"] for n in names]
    sns.barplot(x=r2s, y=names, hue=names, palette="crest", legend=False)
    plt.xlabel("R² Score")
    plt.title("Model Comparison — R² on Test Set")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "model_comparison.png"), dpi=150)
    plt.close()

    # Feature importance (for the best tree-based model, if applicable)
    if hasattr(best_pipeline.named_steps["model"], "feature_importances_"):
        feature_names = (
            NUMERIC_FEATURES
            + list(best_pipeline.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES))
        )
        importances = best_pipeline.named_steps["model"].feature_importances_
        imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False).head(15)

        plt.figure(figsize=(8, 6))
        sns.barplot(data=imp_df, x="importance", y="feature", hue="feature", palette="mako", legend=False)
        plt.title(f"Top Feature Importances — {best_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=150)
        plt.close()

    print("\n" + "=" * 60)
    print(f"BEST MODEL: {best_name}  (R2 = {best_r2:.4f})")
    print("=" * 60)

    joblib.dump(best_pipeline, os.path.join(MODELS_DIR, "best_model.pkl"))
    with open(os.path.join(BASE_DIR, "results.json"), "w") as f:
        json.dump({"results": results, "best_model": best_name, "best_r2": round(float(best_r2), 4)}, f, indent=2)

    return results, best_name


def main():
    df = load_and_engineer()
    explore(df)
    train_and_evaluate(df)
    print("\nAll plots saved to ./plots/  |  Trained model saved to ./models/best_model.pkl")


if __name__ == "__main__":
    main()

"""
Generates a realistic used/new car pricing dataset in the shape commonly
used for this task (similar to the classic Kaggle "Car Price Prediction"
/ UCI Automobile datasets): brand, horsepower, mileage/mpg, engine size,
fuel type, transmission, year, and other specs, with price computed from
a genuine (noisy) pricing model so the relationships are learnable.

The CodeAlpha task PDF's "DOWNLOAD DATASET FROM here" link is a bare
placeholder with no resolvable URL, so this script builds a realistic
stand-in. Swap in a real dataset (e.g. Kaggle's CarPrice_Assignment.csv)
at data/car_data.csv with matching column names to use real data instead
— no code changes needed as long as the target column stays "price".
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 1000

# Brand "goodwill" multiplier — luxury/prestige brands command a premium
brands = {
    "Toyota": 1.00, "Honda": 1.00, "Hyundai": 0.92, "Ford": 0.95,
    "Chevrolet": 0.90, "Volkswagen": 1.05, "Nissan": 0.93,
    "BMW": 1.55, "Mercedes-Benz": 1.60, "Audi": 1.50,
    "Kia": 0.88, "Mazda": 0.98, "Subaru": 1.02, "Lexus": 1.45,
    "Tesla": 1.65, "Jeep": 1.10, "Volvo": 1.30, "Porsche": 2.10,
}
brand_names = list(brands.keys())
fuel_types = ["Petrol", "Diesel", "Hybrid", "Electric"]
transmissions = ["Manual", "Automatic"]

rows = []
for _ in range(N):
    brand = np.random.choice(brand_names)
    goodwill = brands[brand]

    year = np.random.randint(2005, 2025)
    age = 2025 - year

    fuel = np.random.choice(fuel_types, p=[0.55, 0.20, 0.15, 0.10])
    transmission = np.random.choice(transmissions, p=[0.35, 0.65])

    horsepower = np.clip(np.random.normal(180, 60), 70, 650)
    engine_size = np.clip(np.random.normal(2.2, 0.9), 1.0, 6.5)
    mileage_kmpl = np.clip(np.random.normal(15, 5) if fuel != "Electric" else 0, 0, 35)
    odometer_km = max(0, int(np.random.normal(age * 14000, 8000)))

    # --- price model (this is the "real" relationship the ML models learn) ---
    base_price = 18000
    price = base_price * goodwill
    price += horsepower * 55
    price += engine_size * 1800
    price -= age * 900
    price -= odometer_km * 0.03
    price += {"Automatic": 1500, "Manual": 0}[transmission]
    price += {"Petrol": 0, "Diesel": 800, "Hybrid": 2500, "Electric": 5000}[fuel]
    price += np.random.normal(0, 1200)  # market noise
    price = max(1500, price)

    rows.append({
        "brand": brand,
        "year": year,
        "fuel_type": fuel,
        "transmission": transmission,
        "horsepower": round(horsepower, 1),
        "engine_size_l": round(engine_size, 2),
        "mileage_kmpl": round(mileage_kmpl, 1),
        "odometer_km": odometer_km,
        "price": round(price, 2),
    })

df = pd.DataFrame(rows)
df.to_csv("data/car_data.csv", index=False)
print(df.shape)
print(df.head())

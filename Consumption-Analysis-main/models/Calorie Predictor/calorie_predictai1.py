import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
from pathlib import Path


# Resolve CSV path relative to this script to avoid FileNotFoundError
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_filename = "MyFoodData Nutrition Facts SpreadSheet Release 1.4 - SR Legacy and FNDDS.csv"
csv_path = os.path.join(script_dir, csv_filename)

if not os.path.exists(csv_path):
	raise FileNotFoundError(f"Could not find CSV at {csv_path}. Make sure the CSV is located in the same folder as this script.")

df = pd.read_csv(csv_path)
print(df.head())

required_cols = ['Protein (g)', 'Fat (g)', 'Carbohydrate (g)', 'Calories']

for c in required_cols:
	df[c] = pd.to_numeric(df[c], errors='coerce')



y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n Results:")
print(f"   MAE (Average Error): {mae:.2f} calories")
print(f"   Rsqr Score (How good?): {r2:.2f} (closer to 1.0 = better!)")

models_dir = os.path.join(script_dir, "models")
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, "calorie_model.pkl")
joblib.dump(model, model_path)
print(f" Model saved -> {model_path}")


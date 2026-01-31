import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
from pathlib import Path


def load_and_prepare(csv_path: Path):
	if not csv_path.exists():
		raise FileNotFoundError(f"Nutrition CSV not found: {csv_path}")

	df = pd.read_csv(csv_path)

	# Basic diagnostics
	print("Raw rows:", len(df))
	print(df.head())

	# Expecting columns: age, weight, height, gender, activity, calories
	expected = ['age', 'weight', 'height', 'gender', 'activity', 'calories']
	missing = [c for c in expected if c not in df.columns]
	if missing:
		raise KeyError(f"Missing expected columns: {missing}")

	# Coerce numeric columns
	for c in ['age', 'weight', 'height', 'calories']:
		df[c] = pd.to_numeric(df[c], errors='coerce')

	# Drop rows with missing numeric values
	before = len(df)
	df = df.dropna(subset=['age', 'weight', 'height', 'calories']).reset_index(drop=True)
	after = len(df)
	print(f"Dropped {before-after} rows missing numeric values; {after} rows remain.")

	
	df = pd.get_dummies(df, columns=['gender', 'activity'], drop_first=True)

	X = df.drop(columns=['calories'])
	y = df['calories']

	return X, y


def train_and_save_model(X, y, out_path: Path):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	print(f"\nTeaching AI with {len(X_train)} rows")
	print(f"Testing AI with {len(X_test)} rows")

	model = LinearRegression()
	model.fit(X_train, y_train)

	predictions = model.predict(X_test)

	mae = mean_absolute_error(y_test, predictions)
	r2 = r2_score(y_test, predictions)

	print("\nResults:")
	print(f"  Average Error: {mae:.1f} calories")
	print(f"  Smart Score (R2): {r2:.4f}")

	os.makedirs(out_path.parent, exist_ok=True)
	joblib.dump(model, str(out_path))
	print(f"Model saved -> {out_path}")


if __name__ == "__main__":
	BASE = Path(__file__).resolve().parent
	csv_path = BASE / 'nutrition_data.csv'
	# save next to the project's top-level `models` folder (same place calorie model is stored)
	out_model = BASE.parent / 'nutrition_ai_model.pkl'

	try:
		X, y = load_and_prepare(csv_path)
		train_and_save_model(X, y, out_model)
	except Exception as e:
		print("Error during training:")
		raise

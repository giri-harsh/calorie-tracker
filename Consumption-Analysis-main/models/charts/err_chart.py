import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

print("Loading data for error chart...")

# Resolve paths relative to the models folder
BASE = Path(__file__).resolve().parents[1]  # models/
processed_path = BASE / "processed.csv"
myfood_path = BASE / "Food Classifier" / "MyFoodData Nutrition Facts SpreadSheet Release 1.4 - SR Legacy and FNDDS.csv"

if processed_path.exists():
    df = pd.read_csv(processed_path)
    # Expecting normalized column names
    for col in ['protein_g', 'fat_g', 'carbs_g', 'calories']:
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' in {processed_path} but it's missing.")
else:
    # fallback: read the original MyFoodData CSV (header is on row 4 in the raw file)
    if not myfood_path.exists():
        raise FileNotFoundError(f"Neither processed.csv nor MyFoodData CSV found. Checked: {processed_path} and {myfood_path}")
    df = pd.read_csv(myfood_path, header=3, low_memory=False)
    # Map/normalize columns to the short names used by the charts
    mapping = {
        'Protein (g)': 'protein_g',
        'Fat (g)': 'fat_g',
        'Carbohydrate (g)': 'carbs_g',
        'Calories': 'calories'
    }
    # Coerce and create normalized columns
    for src, dst in mapping.items():
        if src in df.columns:
            df[dst] = pd.to_numeric(df[src], errors='coerce')
        else:
            # If some expected verbose column is missing, create dst filled with NaN
            df[dst] = pd.NA

# Drop rows missing required numeric data
before = len(df)
df = df.dropna(subset=['protein_g', 'fat_g', 'carbs_g', 'calories']).reset_index(drop=True)
after = len(df)
print(f"Loaded {after} rows (dropped {before-after} rows with missing values)")

# Prepare X,y and split
X = df[['protein_g', 'fat_g', 'carbs_g']]
y = df['calories']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load trained model
model_path = BASE / "calorie_model.pkl"
if not model_path.exists():
    # also try project-root relative path for backwards compatibility
    alt = Path.cwd() / "models" / "calorie_model.pkl"
    if alt.exists():
        model_path = alt
    else:
        raise FileNotFoundError(f"Trained model not found at {model_path} or {alt}")

model = joblib.load(model_path)

# The saved model may have been trained with different column names (e.g. 'Protein (g)' etc.).
# Map our normalized column names back to the original names expected by the model when possible.
feature_rename_back = {
    'protein_g': 'Protein (g)',
    'fat_g': 'Fat (g)',
    'carbs_g': 'Carbohydrate (g)'
}
model_input = X_test.copy()
# If model was trained with verbose names, rename columns accordingly
if any(name in getattr(model, 'feature_names_in_', []) for name in feature_rename_back.values()):
    model_input = model_input.rename(columns=feature_rename_back)

# Ensure column order matches whatever the model saw during training if possible
try:
    # scikit-learn 1.0+ stores feature_names_in_
    fn = list(getattr(model, 'feature_names_in_', []))
    if fn:
        model_input = model_input[fn]
except Exception:
    pass

predictions = model.predict(model_input)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"\nMAE: {mae:.2f} calories")
print(f"R2: {r2:.2f}")

# Plot
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=predictions, alpha=0.6, s=50)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('Real Calories', fontsize=12)
plt.ylabel('Predicted Calories', fontsize=12)
plt.title('Predictions vs Reality', fontsize=14, fontweight='bold')

text = f'Average Error: {mae:.2f} calories\nSmart Score: {r2:.2f}'
plt.text(0.05, 0.95, text, transform=plt.gca().transAxes,
         fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.show()

print("Done!")
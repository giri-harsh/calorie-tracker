import pandas as pd
import os

# Robust way to determine script and project root
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
except Exception as e:
    raise RuntimeError(f"Could not determine script/project directory: {e}")

# Safely construct the input data directory path
data_dir_base = os.path.join(project_root, "FoodData_Central_foundation_food_csv_2025-04-24")
data_dir = os.path.join(data_dir_base, "FoodData_Central_foundation_food_csv_2025-04-24") \
    if os.path.isdir(os.path.join(data_dir_base, "FoodData_Central_foundation_food_csv_2025-04-24")) \
    else data_dir_base

def checked_read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find CSV: {path}")
    return pd.read_csv(path)

try:
    food = checked_read_csv(os.path.join(data_dir, "food.csv"))
    food_nutrient = checked_read_csv(os.path.join(data_dir, "food_nutrient.csv"))
    nutrient = checked_read_csv(os.path.join(data_dir, "nutrient.csv"))
except Exception as e:
    raise RuntimeError(f"Error loading CSV data: {e}")

# Defensive: Check required columns
required_nutrient_cols = {'id', 'name'}
required_food_cols = {'fdc_id', 'description'}
required_food_nutrient_cols = {'fdc_id', 'nutrient_id', 'amount'}

for df, name, req_cols in [
    (nutrient, "nutrient", required_nutrient_cols),
    (food, "food", required_food_cols),
    (food_nutrient, "food_nutrient", required_food_nutrient_cols),
]:
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns {missing} in {name}.csv")

# merge nutrient names into food_nutrient
merged = food_nutrient.merge(nutrient[['id', 'name']], left_on='nutrient_id', right_on='id', how='left')

# after merge, ensure we still have food and nutrient ref columns
if 'fdc_id' not in merged.columns or 'name' not in merged.columns:
    raise KeyError("After merging, required columns missing from merged DataFrame.")

# merge food descriptions
merged = merged.merge(food[['fdc_id', 'description']], on='fdc_id', how='left')

# filter for nutrients we want
nutrients_we_want = ['Protein', 'Total lipid (fat)', 'Carbohydrate, by difference', 'Energy']
filtered = merged[merged['name'].isin(nutrients_we_want)]

# pivot so each nutrient becomes a column
pivot = filtered.pivot_table(
    index=['fdc_id', 'description'],
    columns='name',
    values='amount',
    aggfunc='mean'
).reset_index()

# Defensive: columns may be missing if not present in this dataset, fill with NaN (or warn)
for col in nutrients_we_want:
    if col not in pivot.columns:
        print(f"Warning: Expected nutrient column '{col}' is missing in data. Filling with NaN.")
        pivot[col] = float('nan')

# Rename columns
pivot.rename(columns={
    'Protein': 'protein_g',
    'Total lipid (fat)': 'fat_g',
    'Carbohydrate, by difference': 'carbs_g',
    'Energy': 'calories'
}, inplace=True)

# Save processed data
data_output_dir = os.path.join(project_root, "data")
os.makedirs(data_output_dir, exist_ok=True)

output_path = os.path.join(data_output_dir, "processed.csv")
pivot.to_csv(output_path, index=False)
print(f"Created processed.csv with {len(pivot)} rows at {output_path}")
print(pivot.head())

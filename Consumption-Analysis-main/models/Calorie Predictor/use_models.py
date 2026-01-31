# There are several issues and points for improvement in this code:

# 1. Hardcoded file paths with Windows-style backslashes:
#    The file paths to the CSV and the model are hard-coded and use backslashes (`\`),
#    which can cause problems on non-Windows systems and are inflexible.
#    It's better to construct file paths using `Path` or `os.path` for cross-platform compatibility.

# 2. Inefficient DataFrame search:
#    The code iterates through all rows in the DataFrame using `.iterrows()` to look for a food match.
#    This is inefficient. Instead, you can use vectorized operations to filter rows efficiently.

# 3. Case-sensitive matching on possible missing data:
#    Using `.lower()` on values that could be `NaN` may result in an error.
#    It's safer to fill missing values or check for them.

# 4. Global loading of large assets:
#    The CSV and model are always loaded, even if this module is imported elsewhere and these resources aren't needed.
#    Consider wrapping them under the `if __name__ == "__main__":` guard,
#    or providing lazy-loading functions.

# 5. Ambiguous variable name (BASE_DIR) is not used:
#    `BASE_DIR` is defined but never actually used for path construction.

# 6. Poor error handling and unclear user feedback:
#    If the food is not found, it just prints "'food' not found." with no suggestions or feedback about available food names.

# 7. Inconsistent spelling: 'protein' is written as 'protien' in other file contexts, but is 'protein_g' in the DataFrame here.

# 8. No input sanitization (besides `.lower()`) or fuzzy matching:
#    Only exact matches work, so if the user makes a typo or uses naming variation, nothing is found.
#    Could consider partial or fuzzy matching.

# Here is a revised (but not executed) version demonstrating improvements for the key issues:
import argparse
import json
from __future__ import annotations

from pathlib import Path
from functools import lru_cache

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "processed.csv"
MODEL_PATH = BASE_DIR / "models" / "calorie_model.pkl"


@lru_cache(maxsize=1)
def load_df() -> pd.DataFrame:
    """Load and cache the processed food database."""
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"Error loading data: CSV not found at {CSV_PATH}")
        return pd.DataFrame()
    except Exception as exc:
        print(f"Error loading data: {exc}")
        return pd.DataFrame()

    numeric_cols = ["protein_g", "fat_g", "carbs_g", "calories"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "food_name" not in df.columns:
        # Provide a fallback column so lookups do not crash.
        df["food_name"] = df.get("description", "")

    df["food_name"] = df["food_name"].astype(str).str.strip()
    return df


@lru_cache(maxsize=1)
def load_model():
    """Load and cache the trained calorie prediction model."""
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"Error loading model: file not found at {MODEL_PATH}")
        return None
    except Exception as exc:
        print(f"Error loading model: {exc}")
        return None


def _row_to_payload(row: pd.Series, model) -> dict:
    protein = float(row.get("protein_g") or 0)
    fat = float(row.get("fat_g") or 0)
    carbs = float(row.get("carbs_g") or 0)
    calories = float(row.get("calories") or 0)

    predicted = None
    if model is not None:
        try:
            predicted = float(model.predict([[protein, fat, carbs]])[0])
        except Exception:
            predicted = None

    payload = {
        "food_name": str(row.get("food_name", "")),
        "protein_g": round(protein, 2),
        "fat_g": round(fat, 2),
        "carbs_g": round(carbs, 2),
        "calories": round(calories, 2) if calories else None,
    }
    if predicted is not None:
        payload["calories_ai"] = round(predicted, 2)
    return payload


def get_food_info_with_ai(food_name: str):
    df = load_df()
    model = load_model()

    if df.empty or model is None or not food_name:
        return None

    normalized = food_name.strip().lower()
    mask = df["food_name"].str.lower().str.contains(normalized, na=False)
    if not mask.any():
        return None

    row = df[mask].iloc[0]
    return _row_to_payload(row, model)


def search_foods(query: str | None = None, limit: int = 25) -> list[dict]:
    df = load_df()
    model = load_model()

    if df.empty:
        return []

    subset = df
    if query:
        text = query.strip().lower()
        subset = df[df["food_name"].str.lower().str.contains(text, na=False)]

    if subset.empty:
        return []

    return [_row_to_payload(row, model) for _, row in subset.head(limit).iterrows()]


def _run_cli():
    parser = argparse.ArgumentParser(description="Lookup foods and predict calories using the trained model.")
    parser.add_argument("--food", "-f", help="Return the best match for a specific food name.")
    parser.add_argument("--search", "-s", help="Return a list of foods matching the search text.")
    parser.add_argument("--limit", type=int, default=25, help="Limit for --search results.")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON instead of text.")
    args = parser.parse_args()

    if args.food:
        result = get_food_info_with_ai(args.food)
        if args.json:
            print(json.dumps({"success": result is not None, "data": result}, ensure_ascii=False))
        else:
            if result:
                print("\nFood:", result["food_name"])
                print(f"Protein: {result['protein_g']}g")
                print(f"Fat: {result['fat_g']}g")
                print(f"Carbs: {result['carbs_g']}g")
                if result.get("calories"):
                    print(f"Calories (label): {result['calories']}")
                if result.get("calories_ai") is not None:
                    print(f"Calories (AI): {result['calories_ai']}")
            else:
                print(f"'{args.food}' not found in database.")
        return

    if args.search:
        results = search_foods(args.search, args.limit)
        if args.json:
            print(json.dumps({"count": len(results), "data": results}, ensure_ascii=False))
        else:
            if not results:
                print("No foods matched your search.")
            else:
                for row in results:
                    cal_display = row.get("calories_ai") or row.get("calories") or "?"
                    print(f"- {row['food_name']} ({cal_display} cal)")
        return

    # Interactive fallback
    print("AI Food Lookup System")
    print("=" * 50)
    while True:
        food = input("\nEnter food name (or 'exit'): ").strip()
        if food.lower() == "exit":
            break
        result = get_food_info_with_ai(food)
        if result:
            print(f"\nFood: {result['food_name']}")
            print(f"Protein: {result['protein_g']}g")
            print(f"Fat: {result['fat_g']}g")
            print(f"Carbs: {result['carbs_g']}g")
            if result.get("calories"):
                print(f"Calories (label): {result['calories']}")
            if result.get("calories_ai") is not None:
                print(f"Calories (AI): {result['calories_ai']}")
        else:
            print(f"'{food}' not found. Try another name.")


if __name__ == "__main__":
    _run_cli()

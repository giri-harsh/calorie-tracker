import pandas as pd
import joblib
from pathlib import Path

# Set up correct paths using pathlib for better cross-platform compatibility
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / 'calorie_model.pkl'
CSV_PATH = BASE_DIR / 'processed.csv'

# Try to load the model safely
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Try to load the food DataFrame safely
try:
    df = pd.read_csv(CSV_PATH)
except Exception as e:
    print(f"Error loading food database: {e}")
    df = pd.DataFrame()

# Your daily calorie goal
daily_goal = 2500
protein_goal = 110  # Fixed spelling

# Start with zero
total_calories = 0
total_protein = 0
total_fat = 0
total_carbs = 0

print("=" * 60)
print("FOOD LOGGER - Track what you eat today")
print("=" * 60)
print(f"Your daily goal: {daily_goal} calories")
print("Type food names. Type 'exit' when done.")
print("=" * 60)

# Loop only if DataFrame loaded successfully
if df.empty:
    print("ERROR: No food data loaded. Exiting.")
else:
    while True:
        food = input("\nEnter food name (or 'exit'): ").strip().lower()
        
        if food == "exit":
            break

        # Defensive: make sure 'food_name' column exists
        if 'food_name' not in df.columns:
            print("Food data error: 'food_name' column missing in database.")
            break

        # Search for the food in the CSV, handle NaN values
        matches = df[
            df['food_name'].astype(str).str.lower().str.contains(food, na=False)
        ]

        if matches.empty:
            print("Food not found in database. Try another name.")
            continue

        # Get the first matching food
        item = matches.iloc[0]

        # Defensive: check required columns exist
        for col in ['protein_g', 'fat_g', 'carbs_g', 'calories']:
            if col not in item:
                print(f"Food data error: '{col}' missing for this item.")
                continue

        # Get the numbers, coerce errors to 0
        try:
            protein = float(item.get('protein_g', 0))
        except Exception:
            protein = 0
        try:
            fat = float(item.get('fat_g', 0))
        except Exception:
            fat = 0
        try:
            carbs = float(item.get('carbs_g', 0))
        except Exception:
            carbs = 0
        try:
            calories = float(item.get('calories', 0))
        except Exception:
            calories = 0

        # Add to totals
        total_protein += protein
        total_fat += fat
        total_carbs += carbs
        total_calories += calories

        # Show the food details
        print(f"\n{item['food_name']}")
        print(f"  Protein: {protein}g")
        print(f"  Fat: {fat}g")
        print(f"  Carbs: {carbs}g")
        print(f"  Calories: {int(calories)}")

        # Show running total
        print(f"\nRunning Total: {int(total_calories)}/{daily_goal} calories")
        print(f"\nRunning protein: {int(total_protein)}/{protein_goal} grams")

    # Show final totals
    print("\n" + "=" * 60)
    print("FINAL TOTALS FOR TODAY")
    print("=" * 60)
    print(f"Protein:  {int(total_protein)}g")
    print(f"Fat:      {int(total_fat)}g")
    print(f"Carbs:    {int(total_carbs)}g")
    print(f"Calories: {int(total_calories)}")

    remaining = daily_goal - total_calories

    print("\n" + "=" * 60)
    if remaining > 0:
        print(f"You can still eat {int(remaining)} more calories today.")
    elif remaining < 0:
        print(f"You ate {int(abs(remaining))} calories OVER your goal.")
    else:
        print("You hit your goal perfectly!")
    print("=" * 60)
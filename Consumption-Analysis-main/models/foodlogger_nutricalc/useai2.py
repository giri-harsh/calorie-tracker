
import joblib
from pathlib import Path
import pandas as pd
import sys

def load_model_and_features(project_root: Path):
    """Load the trained model and infer the feature schema from training CSV."""
    model_path = project_root / 'models' / 'nutrition_ai_model.pkl'
    csv_path = project_root / 'models' / 'foodlogger_nutricalc' / 'nutrition_data.csv'

    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found: {model_path}. Run nutrition_predicai2.py first.")
    if not csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {csv_path}. Cannot infer feature columns.")

    model = joblib.load(model_path)

    # Recreate training feature columns (same preprocessing as training)
    df = pd.read_csv(csv_path)
    for col in ['age', 'weight', 'height', 'calories']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['age', 'weight', 'height', 'calories']).reset_index(drop=True)
    df = pd.get_dummies(df, columns=['gender', 'activity'], drop_first=True)

    feature_cols = [c for c in df.columns if c != 'calories']
    return model, feature_cols


def prepare_input_row(age, weight, height, gender, activity, feature_cols):
    # Build a DataFrame for one sample and get dummies to match training
    tmp = pd.DataFrame([{
        'age': age,
        'weight': weight,
        'height': height,
        'gender': gender,
        'activity': activity
    }])
    tmp = pd.get_dummies(tmp, columns=['gender', 'activity'], drop_first=True)
    # Reindex to the provided feature_cols, filling missing dummy cols with 0
    tmp = tmp.reindex(columns=feature_cols, fill_value=0)
    return tmp

def prompt_int(prompt, min_val=None, max_val=None):
    while True:
        try:
            v = int(input(prompt))
            if min_val is not None and v < min_val:
                print(f"Enter a value >= {min_val}.")
                continue
            if max_val is not None and v > max_val:
                print(f"Enter a value <= {max_val}.")
                continue
            return v
        except ValueError:
            print("Please enter an integer.")

def prompt_float(prompt, min_val=None, max_val=None):
    while True:
        try:
            v = float(input(prompt))
            if min_val is not None and v < min_val:
                print(f"Enter a value >= {min_val}.")
                continue
            if max_val is not None and v > max_val:
                print(f"Enter a value <= {max_val}.")
                continue
            return v
        except ValueError:
            print("Please enter a number.")

def main():
    project_root = Path(__file__).resolve().parents[1]
    try:
        model, feature_cols = load_model_and_features(project_root)
    except Exception as e:
        print("Error loading model/features:", e)
        sys.exit(1)

    print("=" * 60)
    print("AI PHYSIQUE GOAL CALCULATOR - SCIENTIFIC VERSION")
    print("=" * 60)

    print("\nThis calculator uses scientifically validated methods.")
    print("Based on research from WHO, ACSM, and ISSN.")

    print("\n" + "-" * 60)
    print("YOUR CURRENT STATS")
    print("-" * 60)

    age = prompt_int("Age (years): ", min_val=10, max_val=120)
    current_weight = prompt_float("Current weight (kg): ", min_val=20, max_val=500)
    height = prompt_float("Height (cm): ", min_val=50, max_val=250)

    print("\nGender:")
    print("  1 = Male")
    print("  0 = Female")
    gender = prompt_int("Enter: ", min_val=0, max_val=1)

    print("\nActivity level:")
    print("  1 = Sedentary (desk job, no exercise)")
    print("  2 = Light (exercise 1-3 days/week)")
    print("  3 = Moderate (exercise 3-5 days/week)")
    print("  4 = Active (exercise 6-7 days/week)")
    print("  5 = Very active (athlete, physical job)")
    activity = prompt_int("Enter: ", min_val=1, max_val=5)

    print("\n" + "-" * 60)
    print("YOUR GOAL")
    print("-" * 60)

    print("\nWhat do you want to achieve?")
    print("  1 = Lose fat")
    print("  2 = Maintain weight")
    print("  3 = Gain muscle")
    goal_type = prompt_int("Enter: ", min_val=1, max_val=3)

    target_weight = prompt_float("\nTarget weight (kg): ", min_val=10, max_val=500)

    print("\n" + "-" * 60)
    print("CALCULATING SCIENTIFICALLY...")
    print("-" * 60)

    X_row = prepare_input_row(age, current_weight, height, gender, activity, feature_cols)
    try:
        maintenance_calories = int(model.predict(X_row)[0])
    except Exception as e:
        print("Error running model prediction:", e)
        sys.exit(1)

    print(f"\nYour maintenance calories: {maintenance_calories} kcal/day")
    print("(This is the amount you burn daily)")

    weight_difference = target_weight - current_weight

    # These constants should be used where actually needed
    FAT_CALORIES_PER_KG = 7700
    MUSCLE_CALORIES_PER_KG = 3500

    if goal_type == 1:
        weekly_change_rate = 0.005
        weekly_kg_change = current_weight * weekly_change_rate
        daily_calorie_adjustment = -500
        goal_name = "FAT LOSS"
    elif goal_type == 3:
        weekly_change_rate = 0.003
        weekly_kg_change = current_weight * weekly_change_rate
        daily_calorie_adjustment = 300
        goal_name = "MUSCLE GAIN"
    else:
        weekly_kg_change = 0
        daily_calorie_adjustment = 0
        goal_name = "MAINTENANCE"

    target_calories = maintenance_calories + daily_calorie_adjustment

    if weekly_kg_change != 0:
        weeks_to_goal = abs(weight_difference) / weekly_kg_change
        months_to_goal = weeks_to_goal / 4.33
    else:
        weeks_to_goal = 0
        months_to_goal = 0

    if goal_type == 1:
        protein_grams = current_weight * 2.2
    elif goal_type == 3:
        protein_grams = current_weight * 1.8
    else:
        protein_grams = current_weight * 1.6

    fat_calories = target_calories * 0.25
    fat_grams = fat_calories / 9
    protein_calories = protein_grams * 4
    remaining_calories = target_calories - protein_calories - fat_calories
    carbs_grams = max(0, remaining_calories / 4)

    print("\n" + "=" * 60)
    print(f"YOUR {goal_name} PLAN")
    print("=" * 60)

    print(f"\nCurrent stats:")
    print(f"  Weight: {current_weight:.1f} kg")
    print(f"  Target: {target_weight:.1f} kg")
    print(f"  Change needed: {abs(weight_difference):.1f} kg")

    if weeks_to_goal > 0:
        print(f"\nTimeline:")
        print(f"  Weekly change: {weekly_kg_change:.2f} kg/week")
        print(f"  Time to goal: {int(round(weeks_to_goal))} weeks")
        print(f"  That is approximately: {int(round(months_to_goal))} months")

    print("\n" + "-" * 60)
    print("DAILY NUTRITION TARGET")
    print("-" * 60)

    print(f"\nCalories: {int(round(target_calories))} kcal/day")
    print(f"Protein:  {int(round(protein_grams))} grams/day")
    print(f"Fat:      {int(round(fat_grams))} grams/day")
    print(f"Carbs:    {int(round(carbs_grams))} grams/day")

    print("\n" + "-" * 60)
    print("SUGGESTED MEAL DISTRIBUTION")
    print("-" * 60)

    meals = [("Breakfast", 0.25), ("Lunch", 0.35), ("Dinner", 0.30), ("Snacks", 0.10)]
    for meal, pct in meals:
        meal_calories = int(round(target_calories * pct))
        meal_protein = int(round(protein_grams * pct))
        print(f"\n{meal}:")
        print(f"  Calories: {meal_calories} kcal")
        print(f"  Protein: {meal_protein}g")

    print("\n" + "=" * 60)
    print("IMPORTANT SAFETY NOTES")
    print("=" * 60)
    if goal_type == 1 and abs(weight_difference) / max(1, current_weight) > 0.15:
        print("\nWARNING: You are planning significant weight loss. Consult a healthcare professional.")

    print("\nCalculation complete.")

if __name__ == "__main__":
    main()
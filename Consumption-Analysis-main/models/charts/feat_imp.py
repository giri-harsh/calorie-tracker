import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load robot
model = joblib.load("models\Calorie Predictor\calorie_model.pkl")

# Get the coefficients (how important each thing is)
feature_names = ['Protein', 'Fat', 'Carbs']
importance = model.coef_

# Draw bar chart
plt.figure(figsize=(8, 6))
bars = plt.bar(feature_names, importance, color=['red', 'orange', 'blue'])
plt.xlabel('Nutrient')
plt.ylabel('Importance (calories per gram)')
plt.title('Which Nutrient Affects Calories Most?')
plt.grid(True, alpha=0.3, axis='y')

# Add numbers on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}',
             ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.show()

print("\nWhat these numbers mean:")
print(f"1 gram of Protein = {importance[0]:.1f} calories")
print(f"1 gram of Fat = {importance[1]:.1f} calories")
print(f"1 gram of Carbs = {importance[2]:.1f} calories")
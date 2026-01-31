import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv(r"models\processed.csv")
X = df[['protein_g', 'fat_g', 'carbs_g']]
y = df['calories']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load robot
model = joblib.load("models/calorie_model.pkl")

# Get predictions
predictions = model.predict(X_test)

# Calculate errors
errors = y_test - predictions

# Draw histogram
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=20, color='green', alpha=0.7, edgecolor='black')
plt.xlabel('Error (calories)')
plt.ylabel('How Many Times')
plt.title('Distribution of Errors')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Perfect (zero error)')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()
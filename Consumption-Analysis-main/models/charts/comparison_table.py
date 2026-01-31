import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score

print("Loading data...")

# Load 120 people data
df = pd.read_csv(r"models\nutrition_data.csv")

print(f"Loaded {len(df)} people")

# Split data
X = df[['age', 'weight', 'height', 'gender', 'activity']]
y = df['calories']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Testing on {len(X_test)} people")

# Load Model 2
model = joblib.load("models/nutrition_ai_model.pkl")

print("Model 2 loaded!")

# Predict for test people
predictions = model.predict(X_test)

# Calculate errors
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n" + "="*50)
print("MODEL 2 ACCURACY")
print("="*50)
print(f"Average Error: {mae:.1f} calories")
print(f"Smart Score: {r2:.4f}")
print("="*50)

# Show first 10 people
print("\nFirst 10 People:")
for i in range(10):
    real = y_test.iloc[i]
    pred = predictions[i]
    error = abs(real - pred)
    print(f"Person {i+1}: Real={real}, AI={pred:.0f}, Error={error:.0f}")

# CHART 1: Table
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.axis('tight')
ax1.axis('off')

table_data = [['Real', 'AI Predicted', 'Error']]

for i in range(10):
    real = y_test.iloc[i]
    pred = predictions[i]
    error = abs(real - pred)
    table_data.append([f"{real}", f"{pred:.0f}", f"{error:.0f}"])

table = ax1.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Color header
for i in range(3):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')

plt.title('Model 2: First 10 People Comparison', fontsize=14, fontweight='bold', pad=20)

# CHART 2: Scatter plot
fig2, ax2 = plt.subplots(figsize=(10, 6))

sns.scatterplot(x=y_test, y=predictions, alpha=0.6, s=50, ax=ax2)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Predicted')

ax2.set_xlabel('Real Daily Need', fontsize=12)
ax2.set_ylabel('Predicted Need', fontsize=12)
ax2.set_title('Model 2 Accuracy: Real vs Predicted', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# CHART 3: Error bars
fig3, ax3 = plt.subplots(figsize=(10, 6))

errors = [abs(y_test.iloc[i] - predictions[i]) for i in range(len(y_test))]

ax3.bar(range(len(errors)), errors, color='coral', alpha=0.7)
ax3.axhline(y=mae, color='red', linestyle='--', linewidth=2, label=f'Average Error = {mae:.0f}')
ax3.set_xlabel('Person Number', fontsize=12)
ax3.set_ylabel('Error (calories)', fontsize=12)
ax3.set_title('Model 2: Error for Each Person', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nDone! 3 charts shown.")
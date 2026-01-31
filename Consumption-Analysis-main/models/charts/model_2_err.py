import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv(r'models\foodlogger_nutricalc\nutrition_data.csv')
X = df[['age', 'weight', 'height', 'gender', 'activity']]
y = df['calories']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = joblib.load(r'models\foodlogger_nutricalc\nutrion_predicai2.py')


predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Average Error: {mae:.1f} calories")
print(f"Smart Score: {r2:.4f}")

for i in range(5):
    real = y_test.iloc[i]
    pred = predictions[i]
    error = abs(real - pred)
    print(f"\nPerson {i+1}: Real={real}, Predicted={pred:.0f}, Error={error:.0f}")
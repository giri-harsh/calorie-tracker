import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv(r"models\processed.csv")
X = df[['protein_g', 'fat_g', 'carbs_g']]
y = df['calories']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train with different amounts of data
train_sizes = [10, 20, 50, 100, len(X_train)]
train_errors = []
test_errors = []

for size in train_sizes:
    model = LinearRegression()
    model.fit(X_train[:size], y_train[:size])
    
    train_pred = model.predict(X_train[:size])
    test_pred = model.predict(X_test)
    
    train_errors.append(np.sqrt(mean_squared_error(y_train[:size], train_pred)))
    test_errors.append(np.sqrt(mean_squared_error(y_test, test_pred)))

# Draw the chart
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_errors, marker='o', label='Training Error', linewidth=2)
plt.plot(train_sizes, test_errors, marker='s', label='Test Error', linewidth=2)
plt.xlabel('Number of Training Examples')
plt.ylabel('Error (RMSE)')
plt.title('Learning Curve: How Robot Got Smarter')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
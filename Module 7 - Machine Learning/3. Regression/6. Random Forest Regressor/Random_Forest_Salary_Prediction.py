
# Random Forest Regression Teaching Notebook - Salary Prediction (Position Level)

# ==============================================
# 1. Introduction to Random Forest Regression
# ==============================================

"""
Random Forest Regression is an ensemble learning method that builds multiple Decision Trees
and averages their results to improve prediction accuracy and control overfitting.

Each tree is trained on a random subset of the data, and their outputs are averaged.

This makes Random Forest more robust than a single Decision Tree.
"""

# ==============================================
# 2. Importing Libraries
# ==============================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ==============================================
# 3. Loading the Dataset
# ==============================================

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values  # Matrix of features
y = dataset.iloc[:, 2].values    # Target variable

print(dataset.head())

# ==============================================
# 4. Training the Random Forest Regressor
# ==============================================

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(X, y)

# ==============================================
# 5. Predicting a New Result
# ==============================================

y_pred = regressor.predict([[6.5]])
print(f"Predicted salary for level 6.5: {y_pred[0]}")

# ==============================================
# 6. Visualizing Random Forest Regression Results
# ==============================================

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# ==============================================
# ✅ End of Random Forest Regression Notebook
# ==============================================

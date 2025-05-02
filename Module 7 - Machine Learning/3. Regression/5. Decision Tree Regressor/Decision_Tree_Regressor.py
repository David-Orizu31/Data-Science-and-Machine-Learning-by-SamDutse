
# Decision Tree Regression Teaching Notebook - Salary Prediction (Position Level)

# ==============================================
# 1. Introduction to Decision Tree Regression
# ==============================================

"""
Decision Tree Regression models split the data into intervals and fit a constant value for each interval.
The tree splits based on the feature values that reduce error the most.
This results in a step-like prediction curve instead of a smooth one like linear or polynomial regression.
"""

# Visual Intuition:
#
# Salary
#  ^
#  |                ________
#  |               |
#  |      _________|
#  |     |
#  |_____|__________________> Position Level

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
# 4. Training the Decision Tree Regressor
# ==============================================

from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

# ==============================================
# 5. Predicting a New Result
# ==============================================

y_pred = regressor.predict([[6.5]])
print(f"Predicted salary for level 6.5: {y_pred[0]}")

# ==============================================
# 6. Visualizing Decision Tree Regression Results
# ==============================================

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# ==============================================
# ✅ End of Decision Tree Regression Notebook
# ==============================================

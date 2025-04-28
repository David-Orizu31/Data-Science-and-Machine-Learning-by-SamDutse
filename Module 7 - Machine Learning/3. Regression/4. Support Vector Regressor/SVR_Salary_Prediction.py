
# SVR Teaching Notebook - Salary Prediction (Position Level)

# ==============================================
# 1. Introduction to SVR
# ==============================================

"""
Support Vector Regression (SVR) tries to fit the best curve within a margin around the actual values.
Instead of minimizing the overall error (like linear regression), SVR focuses on staying within a margin of tolerance.
The most important data points that define this curve are called Support Vectors.
"""

# Visual Intuition:
#
# |                             o                   o
# |                  o
# |      o                                        o
# |-------------------- Best Fit Curve --------------------
# |         o             o         o
# |
# |
# (Margin above and below the curve)

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
y = dataset.iloc[:, 2].values    # Vector of target variable

# View the dataset
print(dataset.head())

# ==============================================
# 4. Feature Scaling
# ==============================================

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
sc_y = StandardScaler()

X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1, 1))

# ==============================================
# 5. Training the SVR Model
# ==============================================

from sklearn.svm import SVR

regressor = SVR(kernel='rbf')
regressor.fit(X, y.ravel())

# ==============================================
# 6. Predicting a New Result
# ==============================================

# Predict salary for position level 6.5
scaled_pred = regressor.predict(sc_X.transform(np.array([[6.5]])))
y_pred = sc_y.inverse_transform(scaled_pred.reshape(-1, 1))

print(f"Predicted salary for level 6.5: {y_pred[0,0]}")

# ==============================================
# 7. Visualizing SVR Results (Basic)
# ==============================================

plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level (scaled)')
plt.ylabel('Salary (scaled)')
plt.show()

# ==============================================
# 8. Visualizing SVR Results (High Resolution)
# ==============================================

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (SVR - High Resolution)')
plt.xlabel('Position Level (scaled)')
plt.ylabel('Salary (scaled)')
plt.show()

# ==============================================
# ✅ End of SVR Teaching Notebook
# ==============================================

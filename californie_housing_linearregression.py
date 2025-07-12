import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

residuals = y_test - y_pred
q1 = np.percentile(residuals, 25)
q3 = np.percentile(residuals, 75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

mask = (residuals >= lower) & (residuals <= upper)
X_test_clean = X_test[mask]
y_test_clean = y_test[mask]
y_pred_clean = y_pred[mask]

print("r-square:", r2_score(y_test_clean, y_pred_clean))
print("mean-square:", mean_squared_error(y_test_clean, y_pred_clean))

res_clean = y_test_clean - y_pred_clean
plt.figure(figsize=(8, 5))
sns.histplot(res_clean, bins=100, kde=True, color='orange')
plt.title("residual distribution")
plt.xlabel("residual")
plt.ylabel("count")
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.6, color="teal")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")
plt.grid(True)
plt.tight_layout()
plt.show()

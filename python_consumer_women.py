import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv("C:/Users/90545/Downloads/Customer Purchasing Behaviors.csv")
# Drop any missing values (if any)
df = df.dropna()

# Features and target
X = df[["age", "annual_income", "purchase_frequency"]]
y = df["loyalty_score"]

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Visualization
plt.xlabel("age")
plt.ylabel("customer loyalty")
plt.title("customer age vs loyalty")
plt.bar(X_train["age"], y_train)
plt.show()

# Random Forest Model
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_mae = mean_absolute_error(rf_preds, y_test)

# Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
lr_mae = mean_absolute_error(lr_preds, y_test)

# Summary
print(f"Linear Regression MAE : {lr_mae}")
print(f"Random Forest MAE     : {rf_mae}")

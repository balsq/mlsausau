from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import time
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

df= pd.read_csv("data/online_news_popularity_processed.csv")

X = df.drop(' shares', axis=1)
y = df[' shares']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

numeric_cols = [col for col in X_train.columns if len(X_train[col].unique()) > 2 and X_train[col].dtype != 'object']

scaler = StandardScaler()
scaler.fit(X_train[numeric_cols])

X_train[numeric_cols] = scaler.transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

lr_model = LinearRegression()
start_time = time.time()
lr_model.fit(X_train, y_train)
end_time = time.time()
y_pred_lr = lr_model.predict(X_test)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)
training_time_lr = end_time - start_time

print(f"Vreme treniranja Linearne Regresije: {training_time_lr:.4f} sekundi")
print(f"Linear Regression - R^2 Score (Test): {r2_lr:.4f}")
print(f"Linear Regression - RMSE (Test): {rmse_lr:.4f}")


# RIDGE REGRESIJA
alpha_value = 1.0
ridge_model = Ridge(alpha=alpha_value, random_state=42)

start_time = time.time()
ridge_model.fit(X_train, y_train)
end_time = time.time()

training_time_ridge = end_time - start_time

y_pred_ridge = ridge_model.predict(X_test)

rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
r2_ridge = r2_score(y_test, y_pred_ridge)

print("\n--- Performanse Ridge Regresije ---")
print(f"R^2 Score (Test): {r2_ridge:.4f}")
print(f"RMSE (Test): {rmse_ridge:.4f}")
print(f"Vreme treniranja: {training_time_ridge:.4f} sekundi")

##LASO
lasso_model = Lasso(alpha=alpha_value, random_state=42, max_iter=2000)

start_time = time.time()
lasso_model.fit(X_train, y_train)
end_time = time.time()
training_time_lasso = end_time - start_time

y_pred_lasso = lasso_model.predict(X_test)

rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
r2_lasso = r2_score(y_test, y_pred_lasso)

print("\n--- Performanse Lasso Regresije ---")
print(f"R^2 Score (Test): {r2_lasso:.4f}")
print(f"RMSE (Test): {rmse_lasso:.4f}")
print(f"Vreme treniranja: {training_time_lasso:.4f} sekundi")
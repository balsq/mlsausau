import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


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

param_grid = {
    'max_depth': [7, 9, 11],
    'n_estimators': [400, 600],
    'learning_rate': [0.1, 0.05]
}


xgb_base = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1,
    min_child_weight=1
)

grid_search = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    scoring='r2',
    cv=5,
    verbose=2
)

grid_search.fit(X_train, y_train)

print("Najbolji R^2 skor (Cross-Validation): ", grid_search.best_score_)
print("Najbolji parametri: ", grid_search.best_params_)
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("data/online_news_popularity_processed.csv")

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
    "n_estimators": [300, 500, 800],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "reg_alpha": [0, 0.1, 0.5, 1],
    "reg_lambda": [0.5, 1, 2],
    "gamma": [0, 1]
}

model = XGBRegressor(random_state=42)

search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=30,  # više iteracija
    cv=5,
    scoring="r2",
    n_jobs=-1,
    random_state=42
)

search.fit(X_train, y_train)

print("Najbolji parametri:", search.best_params_)
print("Najbolji R² (train CV):", search.best_score_)

# Evaluacija na test setu
best_model = search.best_estimator_
test_r2 = best_model.score(X_test, y_test)
print("R² na test setu:", test_r2)
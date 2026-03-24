from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



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


# --- PODEŠAVANJE: POVEĆANA KOMPLEKSNOST ---
rfr_model_v2 = RandomForestRegressor(
    n_estimators=250,        # Povećan broj stabala za bolju stabilnost
    max_depth=25,            # POVEĆANA DUBLJINA (sa 15 na 25)
    random_state=42,
    n_jobs=-1
)

# --- MERENJE VREMENA ---
print("Treniranje Random Forest modela (v2: Dubina 25, 250 Stabala)...")
start_time = time.time()

rfr_model_v2.fit(X_train, y_train)

end_time = time.time()
training_time_rfr_v2 = end_time - start_time

# --- EVALUACIJA ---
y_pred_rfr_v2 = rfr_model_v2.predict(X_test)
rmse_rfr_v2 = np.sqrt(mean_squared_error(y_test, y_pred_rfr_v2))
r2_rfr_v2 = r2_score(y_test, y_pred_rfr_v2)

print("\n--- Performanse Random Forest Regresora (V2) ---")
print(f"R^2 Score (Test): {r2_rfr_v2:.4f}")
print(f"RMSE (Test): {rmse_rfr_v2:.4f}")
print(f"Vreme treniranja: {training_time_rfr_v2:.2f} sekundi")
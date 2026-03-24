from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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


# --- 1. DEFINISANJE HIPERPARAMETARA ZA PRETRAGU ---
# Fokusiramo se na smanjenje kompleksnosti stabla da bismo se borili protiv overfittinga
param_grid = {
    'max_depth': [5, 7, 9],            # Smanjujemo maksimalnu dubinu
    'min_samples_leaf': [10, 20, 30],  # Minimalni broj uzoraka u listu
    'min_samples_split': [20, 30]      # Minimalni broj uzoraka za podelu
}

# --- 2. INICIJALIZACIJA GRID SEARCH-a ---
grid_search = GridSearchCV(
    estimator=DecisionTreeRegressor(random_state=42),
    param_grid=param_grid,
    scoring='r2',           # Cilj: Maksimizovati R2 skor
    cv=5,                   # Koristimo 5-struku unakrsnu validaciju (5 CV setova)
    n_jobs=-1,              # Koristi sve jezgre CPU-a
    verbose=1
)

# --- 3. POKRETANJE I MERENJE VREMENA ---
print("Pokretanje Grid Search-a za Decision Tree...")
start_time = time.time()

# Pokretanje Grid Search-a na Trening Podacima
grid_search.fit(X_train, y_train)

end_time = time.time()
total_time = end_time - start_time

# --- 4. DOBIJANJE NAJBOLJEG MODELA I EVALUACIJA ---

# Najbolji model
best_dt_model = grid_search.best_estimator_

# Finalna Evaluacija na TEST Setu
y_pred_tuned_dt = best_dt_model.predict(X_test)
r2_tuned_dt = r2_score(y_test, y_pred_tuned_dt)
rmse_tuned_dt = np.sqrt(mean_squared_error(y_test, y_pred_tuned_dt))


print("\n--- Decision Tree (Podešen Model) ---")
print(f"Najbolji Hiperparametri: {grid_search.best_params_}")
print(f"R^2 Score na TEST setu: {r2_tuned_dt:.4f}")
print(f"RMSE na TEST setu: {rmse_tuned_dt:.4f}")
print(f"Ukupno vreme izvršenja Grid Search-a: {total_time:.2f} sekundi")
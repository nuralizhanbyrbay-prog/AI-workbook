from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from xgboost import XGBRegressor
from sklearn.cluster import KMeans


data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target 


coords = X[['Latitude', 'Longitude']]
kmeans = KMeans(n_clusters=12, n_init=10, random_state=42) 
X['Cluster'] = kmeans.fit_predict(coords)
X = pd.get_dummies(X, columns=["Cluster"], prefix="dist")


y_log = np.log1p(y) 
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)


def objective(trial):
    xgb_params = {
        'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('xgb_max_depth', 3, 6),
        'learning_rate': trial.suggest_float('xgb_lr', 0.01, 0.1),
        'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0)
    }
    
    rf_params = {
        'n_estimators': trial.suggest_int('rf_n_estimators', 100, 500),
        'max_depth': trial.suggest_int('rf_max_depth', 5, 15)
    }
    
    estimators = [
        ('xgb', XGBRegressor(**xgb_params, random_state=42)),
        ('rf', RandomForestRegressor(**rf_params, random_state=42))
    ]
    
    stack = StackingRegressor(estimators=estimators, final_estimator=RidgeCV())
    score = cross_val_score(stack, X_train, y_train, cv=3, scoring='r2').mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=15) 

print(f"Лучший R2 на кросс-валидации: {study.best_value:.4f}")

best_params = study.best_params

final_estimators = [
    ('xgb', XGBRegressor(
        n_estimators=best_params['xgb_n_estimators'],
        max_depth=best_params['xgb_max_depth'],
        learning_rate=best_params['xgb_lr'],
        subsample=best_params['xgb_subsample'],
        random_state=42)),
    ('rf', RandomForestRegressor(
        n_estimators=best_params['rf_n_estimators'],
        max_depth=best_params['rf_max_depth'],
        random_state=42))
]

final_stack = StackingRegressor(estimators=final_estimators, final_estimator=RidgeCV())
final_stack.fit(X_train, y_train)


y_pred_log = final_stack.predict(X_test) 

y_test_real = np.expm1(y_test)
y_pred_real = np.expm1(y_pred_log)

print("\n--- ИТОГОВЫЕ МЕТРИКИ ПОРТФОЛИО ---")
print(f"R2 Score (Real): {r2_score(y_test_real, y_pred_real):.4f}")
print(f"MAE: ${mean_absolute_error(y_test_real, y_pred_real) * 100000:.2f}") 
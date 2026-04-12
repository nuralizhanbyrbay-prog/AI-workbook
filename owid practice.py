import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_absolute_error, r2_score
from catboost import CatBoostRegressor 

data = pd.read_csv("owid-energy-data (1).csv") 

data = data.dropna(subset="iso_code") 
target = 'primary_energy_consumption'
features = ['country', 'year', 'population', 'gdp', 'fossil_fuel_consumption', 'renewables_consumption']

df_clean = data.dropna(subset=[target])  
X = df_clean[features]
y = df_clean[target] 

X = X.fillna(-999) 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42) 

# Инициализация модели
model = CatBoostRegressor(
    iterations=2000, # можно больше, так как учимся медленнее
    learning_rate=0.02,
    depth=6,
    l2_leaf_reg=10, # заставляем модель быть скромнее
    early_stopping_rounds=100,
    verbose=100
)

# Обучение
# Добавь 'country' в список features
model.fit(X_train, y_train, cat_features=['country'], eval_set=(X_val, y_val))

# Проверка
preds = model.predict(X_val)
print(f"R2 Score: {r2_score(y_val, preds):.4f}")
print(f"MAE: {mean_absolute_error(y_val, preds):.2f}")

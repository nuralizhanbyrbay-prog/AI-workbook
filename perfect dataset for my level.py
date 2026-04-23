import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures 
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline 
from sklearn.linear_model import LassoCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


data = pd.read_csv('https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv')

le = LabelEncoder()
data['smoker'] = le.fit_transform(data['smoker'])
data["sex"] = le.fit_transform(data["sex"])
data = pd.get_dummies(data, columns=["region"])

X = data.drop("charges", axis=1) 
y = data["charges"] 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 


pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()), 
    ("poly", PolynomialFeatures(degree=2)), 
    ("model", LassoCV(alphas=np.logspace(-3, 3, 100), max_iter=20000))
])


model_log = TransformedTargetRegressor(
    regressor=pipeline, 
    func=np.log1p,       
    inverse_func=np.expm1
)


model_log.fit(X_train, y_train)


y_pred_log = model_log.predict(X_test)


r2 = r2_score(y_test, y_pred_log)
mae = mean_absolute_error(y_test, y_pred_log)
mse = mean_squared_error(y_test, y_pred_log) 

print(f"Коэффициент детерминации (R2) с логарифмом: {r2:.4f}")
print(f"Средняя ошибка (MAE) с логарифмом: ${mae:.2f}")
print(f"Средняя ошибка (MSE) с логарифмом: {mse:.2f}")

# gemini ultra-model vs my model 
print("---- gemini ultra-model vs my model----")


import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

# 1. ПОДГОТОВКА И FEATURE ENGINEERING
def prepare_data(df):
    # Копируем, чтобы не испортить оригинал
    df = df.copy()
    
    # One-Hot Encoding и маппинг
    df['sex'] = df['sex'].map({'female': 0, 'male': 1})
    df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})
    
    # Создаем "золотые" фичи (то, что мы нашли через EDA)
    df['is_obese'] = (df['bmi'] >= 30).astype(int)
    df['smoker_obese'] = df['smoker'] * df['is_obese']
    df['age_sq'] = df['age'] ** 2
    
    # Дамми-переменные для регионов
    df = pd.get_dummies(df, columns=['region'], drop_first=True)
    return df

# Загрузка
data = pd.read_csv('https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv')
df_processed = prepare_data(data)

X = df_processed.drop('charges', axis=1)
y = df_processed['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. САМО-ПОДБОР ПАРАМЕТРОВ (OPTUNA)
def objective(trial):
    # Параметры для XGBoost
    xgb_params = {
        'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('xgb_max_depth', 3, 6),
        'learning_rate': trial.suggest_float('xgb_lr', 0.01, 0.1),
        'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0)
    }
    
    # Параметры для Random Forest
    rf_params = {
        'n_estimators': trial.suggest_int('rf_n_estimators', 100, 500),
        'max_depth': trial.suggest_int('rf_max_depth', 5, 15)
    }
    
    # Создаем ансамбль с текущими параметрами
    estimators = [
        ('xgb', XGBRegressor(**xgb_params, random_state=42)),
        ('rf', RandomForestRegressor(**rf_params, random_state=42))
    ]
    
    stack = StackingRegressor(estimators=estimators, final_estimator=RidgeCV())
    
    # Кросс-валидация (чтобы результат был стабильным)
    score = cross_val_score(stack, X_train, y_train, cv=3, scoring='r2').mean()
    return score

# Запуск поиска (сделаем 20 итераций для скорости)
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

print(f"Лучший R2 на кросс-валидации: {study.best_value:.4f}")

# 3. ФИНАЛЬНАЯ МОДЕЛЬ (ПОРТФОЛИО)
best_params = study.best_params

# Собираем модель на лучших параметрах
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

# 4. ПРОВЕРКА НА ТЕСТЕ
y_pred = final_stack.predict(X_test)
print("\n--- ИТОГОВЫЕ МЕТРИКИ ПОРТФОЛИО ---")
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"MAE: ${mean_absolute_error(y_test, y_pred):.2f}")
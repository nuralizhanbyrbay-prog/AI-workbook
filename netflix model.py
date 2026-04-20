import pandas as pd 
import numpy as np 
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score 
from catboost import CatBoostRegressor 
from sklearn.ensemble import RandomForestRegressor, StackingRegressor 
from xgboost import XGBRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
import optuna 

data = pd.read_csv("netflix_titles.csv")

#EDA
# information of data
print(data.info()) 
# values of data 
print(data.describe()) 
# dataset type 
print(data.head()) 
#Missing values 
print(data.isnull().sum())
#All columns 
data.columns = data.columns.str.strip()
print(data.columns) 

# Cleaning and filling data missing values 
data.dropna(subset=["rating", "duration", "date_added"], inplace=True)

data["director"] = data["director"].fillna("No Director recorded")
data["cast"] = data["cast"].fillna("Unknown cast")

imputer = SimpleImputer(strategy="most_frequent") 
data["country"] = imputer.fit_transform(data[["country"]]).ravel()

data['duration_num'] = data['duration'].apply(lambda x: int(x.split(' ')[0]) if pd.notnull(x) else 0)


data['date_added'] = pd.to_datetime(data['date_added'].str.strip())
data['year_added'] = data['date_added'].dt.year

print(data.isnull().sum())

scaler = RobustScaler()


data['duration_scaled'] = scaler.fit_transform(data[['duration_num']]).ravel() 
# one-hot encoding 
data['type_binary'] = data['type'].map({'Movie': 0, 'TV Show': 1})
#label encoding 
le = LabelEncoder()

categorical_cols = ['type', 'country', 'rating']

# 3. Превращаем текст в числа
for col in categorical_cols:
    data[col + '_encoded'] = le.fit_transform(data[col])

print(data[['type', 'type_encoded', 'country', 'country_encoded']].head())

X = data[['type_encoded', 'country_encoded', 'release_year', 'rating_encoded']]
y = data['duration_num']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

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
study.optimize(objective, n_trials=20)

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

# 4. ПРОВЕРКА НА ТЕСТЕ
y_pred = final_stack.predict(X_test)
print("\n--- ИТОГОВЫЕ МЕТРИКИ ПОРТФОЛИО ---")
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"MAE: ${mean_absolute_error(y_test, y_pred):.2f}")


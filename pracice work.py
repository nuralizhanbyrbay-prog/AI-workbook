import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LassoCV 
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import StandardScaler, PolynomialFeatures 
from sklearn.pipeline import Pipeline 
from sklearn.metrics import mean_squared_error, r2_score 

# Генерируем фейковые данные для тренировки
np.random.seed(42)
n = 1000
data = {
    'Year': np.random.randint(1995, 2025, n),
    'Mileage': np.random.randint(0, 300000, n),
    'Engine_V': np.random.uniform(1.2, 5.0, n),
    'Power': np.random.randint(80, 450, n),
    'Luxury': np.random.choice([0, 1], n, p=[0.8, 0.2]),
}
df = pd.DataFrame(data)
# Генерируем цену с шумом и зависимостью
df['Price'] = (df['Year'] - 1990) * 500000 + (df['Power'] * 15000) - (df['Mileage'] * 2) + (df['Luxury'] * 5000000)
df['Price'] = df['Price'] + np.random.normal(0, 1000000, n)
df.loc[df['Price'] < 500000, 'Price'] = 800000 # Минимальная цена

train_df = df.iloc[:800, :]
test_df = df.iloc[800:, :].drop('Price', axis=1)

train_df.columns = train_df.columns.str.strip()  
test_df.columns = test_df.columns.str.strip() 

X = train_df.select_dtypes(include=[np.number]).drop(["Price"], axis=1) 
y_log = np.log1p(train_df["Price"])
features = X.columns 

pipeline = Pipeline([ 
    ("imputer", SimpleImputer(strategy="median")), 
    ("scaler", StandardScaler()), 
    ("poly", PolynomialFeatures(degree=2)), 
    ("model", LassoCV(alphas=np.logspace(-3, 3, 100), max_iter=20000))
])

X_train, X_val, y_train_log, y_val_log = train_test_split(X, y_log, test_size=0.2, random_state=42) 
pipeline.fit(X_train, y_train_log) 

val_preds_log = pipeline.predict(X_val)
val_preds = np.expm1(val_preds_log)
y_val_real = np.expm1(y_val_log)

rmse_val = np.sqrt(mean_squared_error(y_val_real, val_preds))
r2_val = r2_score(y_val_real, val_preds)

print("\n--- РЕЗУЛЬТАТЫ С LOG-ТРАНСФОРМАЦИЕЙ ---")
print(f"R2 Score: {r2_val:.4f}")
print(f"RMSE (ошибка в $): {rmse_val:.2f}")
print(f"Лучшая Альфа: {pipeline.named_steps['model'].alpha_:.4f}")

coefs = pipeline.named_steps['model'].coef_
print(f"Всего признаков создано: {len(coefs)}")
print(f"Признаков оставлено Lasso: {np.sum(coefs != 0)}")

X_test_kaggle = test_df[features]

final_preds_log = pipeline.predict(X_test_kaggle)

final_predictions = np.expm1(final_preds_log)
submission = pd.DataFrame({
    "Price": final_predictions
}) 
print(submission) 

# comparing my model vs gemini model 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor

# --- ТВОИ ДАННЫЕ (Оставляем как было) ---
np.random.seed(42)
n = 1000
data = {
    'Year': np.random.randint(1995, 2025, n),
    'Mileage': np.random.randint(0, 300000, n),
    'Engine_V': np.random.uniform(1.2, 5.0, n),
    'Power': np.random.randint(80, 450, n),
    'Luxury': np.random.choice([0, 1], n, p=[0.8, 0.2]),
}
df = pd.DataFrame(data)
df['Price'] = (df['Year'] - 1990) * 500000 + (df['Power'] * 15000) - (df['Mileage'] * 2) + (df['Luxury'] * 5000000)
df['Price'] = df['Price'] + np.random.normal(0, 1000000, n)
df.loc[df['Price'] < 500000, 'Price'] = 800000

train_df = df.iloc[:800, :]
test_df = df.iloc[800:, :].drop('Price', axis=1)

# --- УЛУЧШЕННЫЙ ГАЙД ДЛЯ ОЛИМПИАДЫ ---

# 1. Подготовка X и y (Log-transform для цены)
X = train_df.drop(["Price"], axis=1)
y_log = np.log1p(train_df["Price"])

X_train, X_val, y_train_log, y_val_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

# 2. Обучение CatBoost (Мощный бустинг)
# Мы не используем Scaler и Poly, CatBoost сам разберется
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    verbose=0, # Чтобы не спамил в консоль
    random_seed=42
)

model.fit(X_train, y_train_log)

# 3. Предсказание и ОБРАТНАЯ ТРАНСФОРМАЦИЯ
val_preds_log = model.predict(X_val)
val_preds = np.expm1(val_preds_log) # Возвращаем в тенге
y_val_real = np.expm1(y_val_log)    # Реальные цены в тенге

# 4. Считаем честные метрики
rmse_val = np.sqrt(mean_squared_error(y_val_real, val_preds))
r2_val = r2_score(y_val_real, val_preds)

print("\n--- РЕЗУЛЬТАТЫ CATBOOST (UPGRADED) ---")
print(f"R2 Score: {r2_val:.4f}")
print(f"RMSE (ошибка в тенге): {rmse_val:.2f}")

# 5. Смотрим, что было важно (Feature Importance)
import matplotlib.pyplot as plt
feat_importance = model.get_feature_importance()
pd.Series(feat_importance, index=X.columns).sort_values().plot(kind='barh')
plt.title("Что больше всего влияет на цену?")
plt.show()

# 6. Финальный прогноз
final_preds = np.expm1(model.predict(test_df))
submission = pd.DataFrame({"Price": final_preds})
print("\nПервые 5 предсказаний:")
print(submission.head())
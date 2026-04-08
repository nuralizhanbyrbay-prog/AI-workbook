import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score 
import matplotlib.pyplot as plt 

# 1. Загрузка данных
data = pd.read_csv('Life Expectancy Data.csv')

# 2. Очистка имен колонок (убираем лишние пробелы в названиях)
data.columns = data.columns.str.strip()

# 3. Обработка пропусков (Заполняем медианой, чтобы не терять строки)
data = data.fillna(data.median(numeric_only=True))

# 4. Обработка выбросов (твоя логика с квантилями)
mort_vec = data["Adult Mortality"].quantile(0.99)
data_clean = data[data["Adult Mortality"] < mort_vec]

thin_vec = data["thinness 5-9 years"].quantile(0.99)
data_clean = data_clean[data_clean["thinness 5-9 years"] < thin_vec]

# 5. Подготовка признаков (удалили пробел в 'under-five deaths')
features = [
    'Adult Mortality', 
    'Alcohol', 
    'Hepatitis B', 
    'BMI', 
    'HIV/AIDS', # <--- ДОБАВЬ ЭТО
    'Income composition of resources', 
    'Schooling'
]

X = data_clean[features].values 
y = data_clean["Life expectancy"].values 

# 6. Ручное масштабирование (Исправлено X.max)
X_min, X_max = X.min(axis=0), X.max(axis=0) 
X_scaled = (X - X_min) / (X_max - X_min) 

y_min, y_max = y.min(), y.max() 
y_scaled = (y - y_min) / (y_max - y_min) 

# 7. Перемешивание (Shuffle)
np.random.seed(42) 
indices = np.arange(len(X_scaled)) 
np.random.shuffle(indices) 

X_shuffled = X_scaled[indices] 
y_shuffled = y_scaled[indices] 

# 8. Сплит 80/20
split = int(0.8 * len(X_shuffled)) 
X_train, X_test = X_shuffled[:split], X_shuffled[split:] 
y_train, y_test = y_shuffled[:split], y_shuffled[split:]


# --- Gradient Descent Manual ---
# Инициализация
w = np.random.randn(X_train.shape[1]) * 0.01 # веса (теперь вектор, как и y_train)
b = 0.0
alpha = 0.1 # Learning rate
n_iterations = 5000

for epoch in range(n_iterations): 
    # 1. Предсказание (y_hat = X*w + b)
    preds = X_train.dot(w) + b
    
    # 2. Ошибка
    error = preds - y_train 
    
    # 3. Вычисление градиентов
    dw = (X_train.T.dot(error)) / len(X_train)
    db = np.sum(error) / len(X_train)
    
    # 4. Обновление параметров (Шаг вниз!)
    w = w - alpha * dw
    b = b - alpha * db
    
    # Каждые 500 эпох смотрим результат
    if epoch % 500 == 0: 
        current_mse = np.mean(error**2)
        print(f"Epoch {epoch}: MSE = {current_mse:.6f}")

print("\nОбучение завершено!")

# 9. Обучение модели
model = LinearRegression() 
model.fit(X_train, y_train) 

# 10. Предсказание и оценка
y_pred = model.predict(X_test) 

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (Scaled): {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")

# Бонус: посмотрим на веса признаков
importance = pd.Series(model.coef_, index=features).sort_values(ascending=False)
print("\nЗначимость признаков:")
print(importance)
print("Доступные колонки в твоем файле:")
print(data_clean.columns.tolist())
from sklearn.datasets import fetch_california_housing
import numpy as np 
import pandas as pd 
import sklearn.preprocessing as preprocessing
from sklearn.metrics import r2_score 
import matplotlib.pyplot as plt 

data = fetch_california_housing(as_frame=True)
df = data.frame 
# Exploring the data 
print(df.head()) 
print(df.describe()) 
print(df.info()) 
print(df.isnull().sum())


# CLeaning the data
ave_occ_limit = df["AveOccup"].quantile(0.99)
df_clean = df[df["AveOccup"] < ave_occ_limit] 

ave_room_limit = df["AveRooms"].quantile(0.99) 
df_clean = df_clean[df_clean["AveRooms"] < ave_room_limit] 

population_limit = df["Population"].quantile(0.99) 
df_clean = df_clean[df_clean["Population"] < population_limit] 
print(df_clean.describe())

# Scaling the data 
X = df_clean[['AveOccup', 'AveRooms', 'Population', 'MedInc', 'Latitude', 'Longitude']].values
y = df_clean[["MedHouseVal"]].values

# --- ПРАВИЛЬНОЕ МАСШТАБИРОВАНИЕ (ПОКОЛОНОЧНО) ---
# Находим min и max для каждой колонки отдельно
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_scaled = (X - X_min) / (X_max - X_min)

y_min = y.min()
y_max = y.max()
y_scaled = (y - y_min) / (y_max - y_min)

# --- ПЕРЕМЕШИВАНИЕ (ВАЖНО!) ---
indices = np.arange(len(X_scaled))
np.random.seed(42) # Чтобы результат был повторяемым
np.random.shuffle(indices)

X_shuffled = X_scaled[indices]
y_shuffled = y_scaled[indices]

split = int(0.8 * len(X_shuffled)) 
X_train, X_test = X_shuffled[:split], X_shuffled[split:] 
y_train, y_test = y_shuffled[:split], y_shuffled[split:] 

# --- ОБУЧЕНИЕ ---
n_features = X_train.shape[1]
w = np.random.randn(n_features, 1) * 0.01 # Начинаем не с нуля, а со случайных чисел
b = 0.0 
a = 0.5 # Увеличиваем шаг, раз данные теперь "чистые"
n_iterations = 5000 

for i in range(n_iterations):
    y_preds = np.dot(X_train, w) + b 
    error = y_preds - y_train
    
    # Градиенты
    w_grad = (2 / len(X_train)) * np.dot(X_train.T, error) 
    b_grad = (2 / len(X_train)) * np.sum(error) 
    
    w -= a * w_grad 
    b -= a * b_grad 
    
    if i % 500 == 0:
        mse = np.mean(error**2)
        print(f"Эпоха {i}, MSE: {mse:.4f}")

# Исправленный вывод итогов
print(f"\nИтог: Обучили модель!")
print(f"Веса (w): {np.round(w.flatten(), 3)}") # Выведет все 6 весов
print(f"Смещение (b): {b:.3f}")

# --- ТЕСТ (Экзамен) ---
# Для множественной регрессии тоже используем np.dot
test_preds = np.dot(X_test, w) + b
test_mse = np.mean((test_preds - y_test)**2)
print(f"Ошибка на тесте: {test_mse:.4f}") 

def predict_house_price(custom_data):
    # 1. Превращаем ввод в массив
    custom_data = np.array(custom_data).reshape(1, -1)
    
    # 2. Масштабируем (используем min/max из нашего очищенного df_clean!)
    # Важно: используем те же min и max, на которых училась модель
    X_min = df_clean[['AveOccup', 'AveRooms', 'Population', 'MedInc', 'Latitude', 'Longitude']].min().values
    X_max = df_clean[['AveOccup', 'AveRooms', 'Population', 'MedInc', 'Latitude', 'Longitude']].max().values
    
    custom_scaled = (custom_data - X_min) / (X_max - X_min)
    
    # 3. Делаем прогноз (Матричное умножение)
    price_scaled = np.dot(custom_scaled, w) + b
    
    # 4. Размасштабируем обратно в реальные цены (0.14 - 5.0)
    y_min = df_clean["MedHouseVal"].min()
    y_max = df_clean["MedHouseVal"].max()
    
    real_price = price_scaled * (y_max - y_min) + y_min
    return real_price.item()

# ПРИМЕР: Попробуем предсказать цену для района:
# AveOccup=3, AveRooms=5, Population=1200, MedInc=8.0, Lat=37.8, Long=-122.2
my_house = [3, 5, 1200, 8.0, 37.8, -122.2]
prediction = predict_house_price(my_house)

print(f"R² Score: {r2_score(y_test, test_preds):.4f}")
print(f"Предсказанная стоимость: ${prediction * 100_000:.2f}") 
# Тест 1: Район для богатых (Высокий доход, мало людей в доме)
rich_area = [2, 7, 500, 12.0, 34.0, -118.0] 

# Тест 2: Район попроще (Низкий доход, много людей в доме)
poor_area = [5, 3, 2000, 2.0, 34.0, -118.0]

print(f"Цена в богатом районе: ${predict_house_price(rich_area)*100:.1f}k")
print(f"Цена в обычном районе: ${predict_house_price(poor_area)*100:.1f}k")

plt.figure(figsize=(8, 5))
plt.scatter(y_test, test_preds, alpha=0.2, color='darkcyan')
plt.plot([0, 1], [0, 1], color='red', linestyle='--') # Идеальная линия
plt.xlabel("Реальная цена (масштабированная)")
plt.ylabel("Предсказанная цена (масштабированная)")
plt.title("Сравнение прогноза и реальности")
plt.show()
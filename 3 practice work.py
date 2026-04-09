import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LinearRegression, LogisticRegression 
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix

# 1. Загрузка данных
url = "https://raw.githubusercontent.com/campusx-official/laptop-price-predictor-regression-project/main/laptop_data.csv"
data = pd.read_csv(url) 
data.columns = data.columns.str.strip() 

# 2. Очистка и Feature Engineering
data["Ram"] = data["Ram"].str.replace("GB", "").astype("int32") 
data["Weight"] = data["Weight"].str.replace("kg", "").astype("float32")

data["Touchscreen"] = data["ScreenResolution"].apply(lambda x: 1 if "Touchscreen" in x else 0) 
data["IPS"] = data["ScreenResolution"].apply(lambda x: 1 if "IPS" in x else 0) 
data["CpuBrand"] = data["Cpu"].apply(lambda x: " ".join(x.split()[0:3]))

data["HasSSD"] = data["Memory"].apply(lambda x: 1 if "SSD" in x else 0) 
data["HasHDD"] = data["Memory"].apply(lambda x: 1 if "HDD" in x else 0) 

# Вычисляем PPI (важнейший признак!)
new = data['ScreenResolution'].str.extract(r'(\d+)x(\d+)')
data['X_res'] = new[0].astype('int')
data['Y_res'] = new[1].astype('int')
data['PPI'] = (((data['X_res']**2) + (data['Y_res']**2))**0.5 / data['Inches']).astype('float32')

# --- ВНИМАНИЕ: Добавляем PPI в список колонок ---
df_final = data[['Price', 'Ram', 'Weight', 'Inches', 'PPI', 'Touchscreen', 'IPS', 'HasSSD', 'HasHDD', 'CpuBrand', 'TypeName', 'Company']]

# 3. One-Hot Encoding
df_final = pd.get_dummies(df_final, drop_first=True)

# 4. Подготовка X и y
X = df_final.drop('Price', axis=1)
y_reg = df_final['Price']

# Создаем таргет для классификации (выше 90к)
threshold = 90000
y_clf = (df_final['Price'] > threshold).astype(int)

# 5. Масштабирование (StandardScaler)
# Это нужно, чтобы Ram (8) и Price (100000) были в одном масштабе
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Сплит данных
X_train, X_test, y_train_reg, y_test_reg = train_test_split(X_scaled, y_reg, test_size=0.2, random_state=42)
X_train_c, X_test_c, y_train_clf, y_test_clf = train_test_split(X_scaled, y_clf, test_size=0.2, random_state=42)

# --- ЗАДАЧА №1: РЕГРЕССИЯ ---
reg_model = LinearRegression()
reg_model.fit(X_train, y_train_reg)
y_pred_reg = reg_model.predict(X_test)

# --- ЗАДАЧА №2: КЛАССИФИКАЦИЯ ---
clf_model = LogisticRegression(max_iter=1000)
clf_model.fit(X_train_c, y_train_clf)
y_pred_clf = clf_model.predict(X_test_c)

# 7. Вывод результатов
print("--- РЕЗУЛЬТАТЫ РЕГРЕССИИ ---")
print(f"Средняя ошибка (MAE): {mean_absolute_error(y_test_reg, y_pred_reg):.2f}")

print("\n--- РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ ---")
print(f"Точность (Accuracy): {accuracy_score(y_test_clf, y_pred_clf):.4f}") 
print("Матрица ошибок:")
print(confusion_matrix(y_test_clf, y_pred_clf))

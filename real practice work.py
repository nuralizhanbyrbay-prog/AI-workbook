from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import sklearn.preprocessing as preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = fetch_california_housing(as_frame=True)
df = data.frame

# ==========================================
# 2. Cleaning
# ==========================================
ave_occ_limit = df["AveOccup"].quantile(0.99)
ave_room_limit = df["AveRooms"].quantile(0.99)
population_limit = df["Population"].quantile(0.99)

df_clean = df[
    (df["AveOccup"] < ave_occ_limit) & 
    (df["AveRooms"] < ave_room_limit) & 
    (df["Population"] < population_limit)
]

# ==========================================
# 3.Feature prepararion and scaling
# ==========================================
features = ['AveOccup', 'AveRooms', 'Population', 'MedInc', 'Latitude', 'Longitude']
X = df_clean[features].values
y = df_clean[["MedHouseVal"]].values

# Min-Max Scaling
X_min, X_max = X.min(axis=0), X.max(axis=0)
X_scaled = (X - X_min) / (X_max - X_min)

y_min, y_max = y.min(), y.max()
y_scaled = (y - y_min) / (y_max - y_min)

# mixing 
np.random.seed(42)
indices = np.arange(len(X_scaled))
np.random.shuffle(indices)

X_shuffled = X_scaled[indices]
y_shuffled = y_scaled[indices]

# split train/test 
split = int(0.8 * len(X_shuffled))
X_train, X_test = X_shuffled[:split], X_shuffled[split:]
y_train, y_test = y_shuffled[:split], y_shuffled[split:]

# ==========================================
# 4. handmade linear regression 
# ==========================================
n_features = X_train.shape[1]
w = np.random.randn(n_features, 1) * 0.01
b = 0.0
alpha = 0.5
n_iterations = 5000

print("--- Обучение Линейной Регрессии ---")
for i in range(n_iterations):
    y_preds = np.dot(X_train, w) + b
    error = y_preds - y_train
    
    w_grad = (2 / len(X_train)) * np.dot(X_train.T, error)
    b_grad = (2 / len(X_train)) * np.sum(error)
    
    w -= alpha * w_grad
    b -= alpha * b_grad
    
    if i % 1000 == 0:
        mse = np.mean(error**2)
        print(f"Эпоха {i}, MSE: {mse:.4f}")

# testing lin.reg model 
test_preds_lin = np.dot(X_test, w) + b
lin_r2 = r2_score(y_test, test_preds_lin)

# ==========================================
# 5. Random Forest
# ==========================================
print("\n--- Обучение Random Forest ---")
forest_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
forest_model.fit(X_train, y_train.ravel()) 

forest_preds = forest_model.predict(X_test)
forest_r2 = r2_score(y_test, forest_preds)

# ==========================================
# 6. comparing results
# ==========================================
print("\n" + "="*30)
print(f"R² Линейная регрессия: {lin_r2:.4f}")
print(f"R² Random Forest:      {forest_r2:.4f}")
print("="*30)

# ==========================================
# 7. Function of prediction
# ==========================================
def predict_house_price(custom_data, weights, bias):
    custom_data = np.array(custom_data).reshape(1, -1)
    
    # Масштабирование
    X_min_vals = df_clean[features].min().values
    X_max_vals = df_clean[features].max().values
    custom_scaled = (custom_data - X_min_vals) / (X_max_vals - X_min_vals)
    
    # prediction
    price_scaled = np.dot(custom_scaled, weights) + bias
    
    # Scaling
    y_min_val = df_clean["MedHouseVal"].min()
    y_max_val = df_clean["MedHouseVal"].max()
    real_price = price_scaled * (y_max_val - y_min_val) + y_min_val
    return real_price.item()

# Testing cases
rich_area = [2, 7, 500, 12.0, 34.0, -118.0]
poor_area = [5, 3, 2000, 2.0, 34.0, -118.0]

print(f"\nЦена (Rich) [LinReg]: ${predict_house_price(rich_area, w, b)*100:.1f}k")
print(f"Цена (Poor) [LinReg]: ${predict_house_price(poor_area, w, b)*100:.1f}k")

# ==========================================
# 8. Vizualization
# ==========================================
plt.figure(figsize=(10, 6))
plt.scatter(y_test, test_preds_lin, alpha=0.2, color='darkcyan', label=f'Linear Regression (R²={lin_r2:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Perfect Prediction')
plt.xlabel("Real Price (Scaled)")
plt.ylabel("Predicted Price (Scaled)")
plt.title("NumPy Linear Regression: Actual vs Predicted")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

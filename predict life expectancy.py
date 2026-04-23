import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score 
import matplotlib.pyplot as plt 

data = pd.read_csv('Life Expectancy Data.csv')

data.columns = data.columns.str.strip()

data = data.fillna(data.median(numeric_only=True))

mort_vec = data["Adult Mortality"].quantile(0.99)
data_clean = data[data["Adult Mortality"] < mort_vec]

thin_vec = data["thinness 5-9 years"].quantile(0.99)
data_clean = data_clean[data_clean["thinness 5-9 years"] < thin_vec]

features = [
    'Adult Mortality', 
    'Alcohol', 
    'Hepatitis B', 
    'BMI', 
    'HIV/AIDS', 
    'Income composition of resources', 
    'Schooling'
]

X = data_clean[features].values 
y = data_clean["Life expectancy"].values 


X_min, X_max = X.min(axis=0), X.max(axis=0) 
X_scaled = (X - X_min) / (X_max - X_min) 

y_min, y_max = y.min(), y.max() 
y_scaled = (y - y_min) / (y_max - y_min) 

np.random.seed(42) 
indices = np.arange(len(X_scaled)) 
np.random.shuffle(indices) 

X_shuffled = X_scaled[indices] 
y_shuffled = y_scaled[indices] 


split = int(0.8 * len(X_shuffled)) 
X_train, X_test = X_shuffled[:split], X_shuffled[split:] 
y_train, y_test = y_shuffled[:split], y_shuffled[split:]


w = np.random.randn(X_train.shape[1]) * 0.01 
b = 0.0
alpha = 0.1 
n_iterations = 5000

for epoch in range(n_iterations): 
    preds = X_train.dot(w) + b
    
    error = preds - y_train 
    
    dw = (X_train.T.dot(error)) / len(X_train)
    db = np.sum(error) / len(X_train)
    
    w = w - alpha * dw
    b = b - alpha * db
    
    
    if epoch % 500 == 0: 
        current_mse = np.mean(error**2)
        print(f"Epoch {epoch}: MSE = {current_mse:.6f}")

print("\nОбучение завершено!")


model = LinearRegression() 
model.fit(X_train, y_train) 

y_pred = model.predict(X_test) 
# Testing cases 
long_life_= [1, 0.02, 50, 10, 0.1, 0.8, 15]
short_life_ = [0.8, 0.01, 30, 20, 0.5, 0.4, 10] 
long_life_scaled = (np.array(long_life_) - X_min) / (X_max - X_min)
short_life_scaled = (np.array(short_life_) - X_min) / (X_max - X_min)   

long_life_pred = model.predict(long_life_scaled.reshape(1, -1))
short_life_pred = model.predict(short_life_scaled.reshape(1, -1)) 
print(f"\nPredicted Life Expectancy for long life case: {long_life_pred[0] * (y_max - y_min) + y_min:.2f} years")
print(f"Predicted Life Expectancy for short life case: {short_life_pred[0] * (y_max - y_min) + y_min:.2f} years") 
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (Scaled): {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")

importance = pd.Series(model.coef_, index=features).sort_values(ascending=False)
print("\nЗначимость признаков:")
print(importance)
print("Доступные колонки в твоем файле:")
print(data_clean.columns.tolist())
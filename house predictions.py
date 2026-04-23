import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, PolynomialFeatures 
from sklearn.impute import SimpleImputer 
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline 
from sklearn.metrics import mean_squared_error, r2_score 


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv") 

train.columns = train.columns.str.strip() 
test.columns = test.columns.str.strip() 


X = train.select_dtypes(include=[np.number]).drop(["SalePrice", "Id"], axis=1) 

y_log = np.log1p(train["SalePrice"]) 

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


X_test_kaggle = test[features]

final_preds_log = pipeline.predict(X_test_kaggle)

final_predictions = np.expm1(final_preds_log)

submission = pd.DataFrame({
    "Id": test["Id"],
    "SalePrice": final_predictions
})

submission.to_csv("submission_final.csv", index=False)

print("\n--- ГОТОВО ---")
print("Файл 'submission_final.csv' готов")
#Скрипт для создания и обучения модели машинного обучения.

import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Создаем папку для сохранения модели
os.makedirs('models', exist_ok=True)

# Загружает предобработанные данные и обучает модель.
def train_model():
    print("Обучение модели\n")
    
    # Загружаем предобработанные данные
    print("Загрузка предобработанных данных...")
    X_train = np.load('models/X_train_scaled.npy')
    y_train = np.load('models/y_train.npy')
    feature_info = joblib.load('models/feature_info.pkl')
    
    print(f"Загружено {len(X_train)} образцов")
    print(f"Признаки: {feature_info['features']}\n")
    
    # Создаем и обучаем модель (используем Random Forest для лучшей точности)
    print("Создание модели Random Forest...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    # Обучаем модель
    print("Обучение модели...")
    model.fit(X_train, y_train)
    
    # Делаем предсказания на тренировочных данных для оценки
    y_pred_train = model.predict(X_train)
    
    # Вычисляем метрики
    mse = mean_squared_error(y_train, y_pred_train)
    mae = mean_absolute_error(y_train, y_pred_train)
    r2 = r2_score(y_train, y_pred_train)
    
    print("\nМетрики на тренировочных данных:")
    print(f"   - MSE (среднеквадратичная ошибка): {mse:.2f}")
    print(f"   - MAE (средняя абсолютная ошибка): {mae:.2f}")
    print(f"   - R² (коэффициент детерминации): {r2:.4f}")
    
    # Сохраняем модель
    joblib.dump(model, 'models/model.pkl')
    print("\nМодель сохранена в models/model.pkl")
    
    # Сохраняем информацию о метриках
    metrics = {
        'mse': mse,
        'mae': mae,
        'r2': r2
    }
    joblib.dump(metrics, 'models/training_metrics.pkl')
    
    return model, metrics

if __name__ == "__main__":
    train_model()
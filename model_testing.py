#Скрипт для тестирования модели на тестовых данных.

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Загружает тестовые данные и применяет сохраненный scaler.
def load_and_preprocess_test_data():
    print("Предобработка тестовых данных\n")
    
    # Загружаем сохраненный scaler и информацию о признаках
    scaler = joblib.load('models/scaler.pkl')
    feature_info = joblib.load('models/feature_info.pkl')
    
    all_test_data = []
    
    # Загружаем все тестовые данные
    print("Загрузка тестовых данных...")
    for file in os.listdir('test'):
        if file.endswith('.csv'):
            file_path = os.path.join('test', file)
            df = pd.read_csv(file_path)
            all_test_data.append(df)
            print(f"  - Загружен {file} ({len(df)} записей)")
    
    # Объединяем все тестовые данные
    test_df = pd.concat(all_test_data, ignore_index=True)
    print(f"\nВсего загружено {len(test_df)} тестовых записей")
    
    # Подготавливаем признаки
    features = feature_info['features']
    X_test = test_df[features].values
    y_test = test_df['temperature'].values
    
    # Применяем scaling
    X_test_scaled = scaler.transform(X_test)
    
    return X_test_scaled, y_test, test_df

# Загружает модель и тестирует её на тестовых данных.
def test_model():
    print("Тестирование модели\n")
    
    # Загружаем модель
    print("Загрузка модели...")
    model = joblib.load('models/model.pkl')
    
    # Загружаем и предобрабатываем тестовые данные
    X_test, y_test, test_df = load_and_preprocess_test_data()
    
    # Делаем предсказания
    print("\nВыполнение предсказаний...")
    y_pred = model.predict(X_test)
    
    # Вычисляем метрики
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\nРезультаты тестирования модели:")
    print("=" * 50)
    print(f"   - MSE (среднеквадратичная ошибка): {mse:.2f}")
    print(f"   - MAE (средняя абсолютная ошибка): {mae:.2f}")
    print(f"   - R² (коэффициент детерминации): {r2:.4f}")
    print("=" * 50)
    
    # Сравнение с тренировочными метриками
    training_metrics = joblib.load('models/training_metrics.pkl')
    print("\nСравнение с тренировочными метриками:")
    print(f"   - R² train: {training_metrics['r2']:.4f} | R² test: {r2:.4f}")
    print(f"   - Разница R²: {abs(training_metrics['r2'] - r2):.4f}")
    
    if r2 < 0.7:
        print("\nПредупреждение: Модель показывает низкое качество на тестовых данных!")
    elif r2 > 0.9:
        print("\nОтлично! Модель показывает высокое качество на тестовых данных.")
    else:
        print("\nМодель показывает приемлемое качество на тестовых данных.")
    
    # Сохраняем результаты предсказаний
    results_df = test_df.copy()
    results_df['predicted_temperature'] = y_pred
    results_df['error'] = np.abs(results_df['temperature'] - results_df['predicted_temperature'])
    results_df.to_csv('test/predictions.csv', index=False)
    print("\nРезультаты предсказаний сохранены в test/predictions.csv")
    
    # Сохраняем метрики
    test_metrics = {
        'mse': mse,
        'mae': mae,
        'r2': r2
    }
    joblib.dump(test_metrics, 'models/test_metrics.pkl')
    
    return test_metrics

if __name__ == "__main__":
    test_model()
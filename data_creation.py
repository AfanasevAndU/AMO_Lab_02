# Скрипт для генерации наборов данных с температурой.
# Создает несколько датасетов с трендом, сезонностью, шумом и аномалиями.

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

# Создаем папки train и test, если они не существуют
os.makedirs('train', exist_ok=True)
os.makedirs('test', exist_ok=True)

# Генерирует данные о температуре за указанное количество дней.
# Параметры:
# - days: количество дней
# - trend: линейный тренд (градусов в день)
# - noise_std: стандартное отклонение шума
# - anomalies: добавлять ли аномалии
# - seed: для воспроизводимости
def generate_temperature_data(days, trend=0.01, noise_std=2.0, anomalies=False, seed=42):
    np.random.seed(seed)
    
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(days)]
    
    # Базовые параметры: годовая сезонность + тренд
    base_temp = 15  # средняя температура
    seasonal_amplitude = 10  # амплитуда сезонных колебаний
    
    # Создаем признаки
    day_of_year = np.array([date.timetuple().tm_yday for date in dates])
    
    # Температура с сезонностью (синусоида) и трендом
    seasonal_temp = seasonal_amplitude * np.sin(2 * np.pi * day_of_year / 365)
    trend_temp = trend * np.arange(days)
    temperature = base_temp + seasonal_temp + trend_temp
    
    # Добавляем шум
    noise = np.random.normal(0, noise_std, days)
    temperature += noise
    
    # Добавляем аномалии (резкие скачки температуры)
    if anomalies:
        num_anomalies = int(days * 0.05)  # 5% аномалий
        anomaly_indices = np.random.choice(days, num_anomalies, replace=False)
        for idx in anomaly_indices:
            # Аномалия: резкое повышение или понижение на 10-20 градусов
            temperature[idx] += np.random.choice([-1, 1]) * np.random.uniform(10, 20)
    
    # Создаем DataFrame
    df = pd.DataFrame({
        'date': dates,
        'day_of_year': day_of_year,
        'day': [date.day for date in dates],
        'month': [date.month for date in dates],
        'temperature': temperature
    })
    
    return df

# Генерируем тренировочные данные (3 набора)
print("Генерация тренировочных данных...")

# Датасет 1: базовые данные с небольшим шумом
train_df1 = generate_temperature_data(500, trend=0.005, noise_std=1.5, anomalies=False, seed=42)
train_df1.to_csv('train/train_dataset_1.csv', index=False)
print("  - train_dataset_1.csv создан (500 дней, базовые данные)")

# Датасет 2: данные с большим шумом и аномалиями
train_df2 = generate_temperature_data(500, trend=0.01, noise_std=3.0, anomalies=True, seed=43)
train_df2.to_csv('train/train_dataset_2.csv', index=False)
print("  - train_dataset_2.csv создан (500 дней, шум + аномалии)")

# Датасет 3: данные с сильным трендом
train_df3 = generate_temperature_data(500, trend=0.02, noise_std=1.0, anomalies=False, seed=44)
train_df3.to_csv('train/train_dataset_3.csv', index=False)
print("  - train_dataset_3.csv создан (500 дней, сильный тренд)")

# Генерируем тестовые данные (2 набора)
print("\nГенерация тестовых данных...")

# Датасет 1: стандартные тестовые данные
test_df1 = generate_temperature_data(200, trend=0.005, noise_std=1.5, anomalies=False, seed=45)
test_df1.to_csv('test/test_dataset_1.csv', index=False)
print("  - test_dataset_1.csv создан (200 дней)")

# Датасет 2: данные с аномалиями для проверки устойчивости
test_df2 = generate_temperature_data(200, trend=0.01, noise_std=2.0, anomalies=True, seed=46)
test_df2.to_csv('test/test_dataset_2.csv', index=False)
print("  - test_dataset_2.csv создан (200 дней, с аномалиями)")

print("\nГенерация данных завершена!")
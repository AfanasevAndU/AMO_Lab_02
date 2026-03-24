#Скрипт для предобработки данных с использованием StandardScaler.

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Создаем папку для сохранения препроцессора
os.makedirs('models', exist_ok=True)

#Загружает все CSV файлы из папки, объединяет их и применяет предобработку.
def load_and_preprocess_data(data_folder):
    all_data = []
    
    # Загружаем все CSV файлы из папки
    for file in os.listdir(data_folder):
        if file.endswith('.csv'):
            file_path = os.path.join(data_folder, file)
            df = pd.read_csv(file_path)
            all_data.append(df)
            print(f"  Загружен {file}")
    
    # Объединяем все данные
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nВсего загружено {len(combined_df)} записей")
    
    # Создаем признаки для модели
    features = ['day_of_year', 'day', 'month']
    target = 'temperature'
    
    X = combined_df[features].values
    y = combined_df[target].values
    
    return X, y, features

# Основная функция для предобработки тренировочных данных.
def preprocess_training_data():
    print("Предобработка тренировочных данных\n")
    
    # Загружаем тренировочные данные
    print("Загрузка тренировочных данных...")
    X_train, y_train, features = load_and_preprocess_data('train')
    
    # Применяем StandardScaler к признакам
    print("\nПрименение StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Сохраняем scaler для использования в тестировании
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Scaler сохранен в models/scaler.pkl")
    
    # Сохраняем обработанные данные для обучения
    np.save('models/X_train_scaled.npy', X_train_scaled)
    np.save('models/y_train.npy', y_train)
    
    # Сохраняем информацию о признаках
    feature_info = {'features': features, 'n_samples': len(X_train_scaled)}
    joblib.dump(feature_info, 'models/feature_info.pkl')
    
    print(f"\nПредобработка завершена!")
    print(f"   - Признаков: {X_train_scaled.shape[1]}")
    print(f"   - Образцов: {X_train_scaled.shape[0]}")
    print(f"   - Среднее после scaling: {X_train_scaled.mean(axis=0).round(2)}")
    print(f"   - Стандартное отклонение после scaling: {X_train_scaled.std(axis=0).round(2)}")
    
    return X_train_scaled, y_train

if __name__ == "__main__":
    preprocess_training_data()
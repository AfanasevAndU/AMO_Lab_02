#!/bin/bash
echo "Запуск конвейера машинного обучения"
echo ""

# Функция для проверки успешности выполнения предыдущей команды
check_error() {
    if [ $? -ne 0 ]; then
        echo "Ошибка: $1"
        exit 1
    fi
}

# Проверяем наличие Python
if ! command -v python3 &> /dev/null; then
    echo "Python3 не установлен!"
    exit 1
fi

# Проверяем наличие необходимых библиотек
echo "Проверка зависимостей..."
python -c "import numpy, pandas, sklearn, joblib" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Установка необходимых библиотек..."
    pip3 install numpy pandas scikit-learn joblib
    check_error "Не удалось установить зависимости"
fi

echo ""
echo "Все зависимости установлены"
echo ""

# Этап 1: Создание данных
echo "Этап 1: Создание наборов данных"
python data_creation.py
check_error "Ошибка при создании данных"
echo ""

# Этап 2: Предобработка данных
echo "Этап 2: Предобработка данных"
python model_preprocessing.py
check_error "Ошибка при предобработке данных"
echo ""

# Этап 3: Обучение модели
echo "Этап 3: Обучение модели"
python model_preparation.py
check_error "Ошибка при обучении модели"
echo ""

# Этап 4: Тестирование модели
echo "Этап 4: Тестирование модели"
python model_testing.py
check_error "Ошибка при тестировании модели"
echo ""

# Финальный отчет
echo "Конвейер успешно выполнен!"
echo ""
echo "Результаты работы:"
echo "   - Обучающие данные: ./train/"
echo "   - Тестовые данные: ./test/"
echo "   - Модель и препроцессор: ./models/"
echo "   - Результаты предсказаний: ./test/predictions.csv"
echo ""
echo "Метрики модели сохранены в ./models/test_metrics.pkl"
echo "Нажмите Enter для выхода..."
read
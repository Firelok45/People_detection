# Детекция людей с на видео

Проект для детекции людей на видео с использованием модели YOLOv8.

## Описание
Этот проект выполняет детекцию людей в видеофайле `crowd.mp4`, рисует прямоугольники (bounding boxes) с метками уровня уверенности и сохраняет результат в выходное видео.

## Структура проекта
- `main.py`: Точка входа в программу.
- `src/detector.py`: Модуль для детекции людей с использованием YOLOv8.
- `input_videos/crowd.mp4`: Входное видео.
- `output_videos/output.mp4`: Выходное видео с результатами.
- `requirements.txt`: Список зависимостей.
- `README.md`: Документация.
- `.gitignore`: Файл для исключения ненужных файлов.

## Установка
1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/Firelok45/People_detection
   cd People_detection
2. Создайте и активируйте виртуальное окружение:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/MacOS
   venv\Scripts\activate     # Windows
2. Установите зависимости:
   ```bash
   pip install -r requirements.txt # Устанавливаем зависисмости
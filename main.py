"""
Главный модуль для запуска детекции людей на видео.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from src.detector import PeopleDetector  # Теперь импорт будет работать


# Настройки
CONFIDENCE_THRESHOLD = 0.4 # Уверенность распознавания
MODEL_PATH = "yolov8n.pt" # Используемая модель
INPUT_VIDEO = "input_video"
OUTPUT_VIDEO = "output_video/output.mp4"

if __name__ == "__main__":
    detector = PeopleDetector(MODEL_PATH, CONFIDENCE_THRESHOLD)
    detector.detect(INPUT_VIDEO, OUTPUT_VIDEO)

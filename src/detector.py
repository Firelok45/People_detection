"""
Модуль для детекции людей на видео с использованием YOLOv8.
"""

import os
import cv2
from ultralytics import YOLO

class PeopleDetector:
    """
    Класс для детекции людей в видео с использованием YOLOv8.
    """

    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        """
        Инициализация детектора.

        :param model_path: Путь к модели YOLO (например, yolov8n.pt, yolov8m.pt).
        :param confidence_threshold: Минимальный уровень доверия (0.0 - 1.0).
        """
        # Проверяем, существует ли файл модели локально
        if os.path.exists(model_path):
            print(f"Файл модели '{model_path}' найден локально.")
        else:
            print(f"Файл модели '{model_path}' не найден локально. Ultralytics попытается загрузить его автоматически.")

        print(f"Загрузка модели из: {model_path}...")
        try:
            self.model = YOLO(model_path)
            print("Модель успешно загружена.")
        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}")
            raise
        self.confidence_threshold = confidence_threshold

    def detect(self, input_video: str, output_video: str):
        """
        Обрабатывает видео, выполняя детекцию людей и сохраняя результат.

        :param input_video: Путь к входному видеофайлу.
        :param output_video: Путь для сохранения обработанного видео.
        """
        # Создаем папку для выходных видео, если её нет
        output_dir = os.path.dirname(output_video)
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            print(f"Ошибка: Не удалось открыть видеофайл '{input_video}'.")
            raise ValueError(f"Cannot open video file: {input_video}")

        frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

        print("Начало обработки видео...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame)

            for result in results:
                for box in result.boxes:
                    conf, cls = float(box.conf[0]), int(box.cls[0])

                    if cls == 0 and conf >= self.confidence_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        label = f"Person {conf:.2f}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            out.write(frame)
            cv2.imshow("YOLOv8 Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Обработка прервана пользователем.")
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Обработка завершена. Результат сохранён в '{output_video}'.")
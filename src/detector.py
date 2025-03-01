# """
# Модуль для детекции людей на видео с использованием YOLOv8.
# """
#
# import os
# import cv2
# from ultralytics import YOLO
#
# class PeopleDetector:
#     """
#     Класс для детекции людей в видео с использованием YOLOv8.
#     """
#
#     def __init__(self, model_path: str, confidence_threshold: float):
#         """
#         Инициализация детектора.
#
#         :param model_path: Путь к модели YOLO (например, yolov8n.pt, yolov8m.pt).
#         :param confidence_threshold: Минимальный уровень доверия (0.0 - 1.0).
#         """
#         # Проверяем, существует ли файл модели локально
#         if os.path.exists(model_path):
#             print(f"Файл модели '{model_path}' найден локально.")
#         else:
#             print(f"Файл модели '{model_path}' не найден локально. Ultralytics попытается загрузить его автоматически.")
#
#         print(f"Загрузка модели из: {model_path}")
#         try:
#             self.model = YOLO(model_path)
#             print("Модель успешно загружена.")
#         except Exception as e:
#             print(f"Ошибка при загрузке модели: {e}")
#             raise
#         self.confidence_threshold = confidence_threshold
#
#     @staticmethod
#     def find_video(folder_path):
#         # Получаем список всех файлов в папке
#         for file in os.listdir(folder_path):
#             if file.endswith(".mp4"):
#                 return os.path.join(folder_path, file)
#
#     def detect(self, input_video: str, output_video: str):
#         """
#         Обрабатывает видео, выполняя детекцию людей и сохраняя результат.
#
#         :param input_video: Путь к входному видеофайлу.
#         :param output_video: Путь для сохранения обработанного видео.
#         """
#         # Создаем папку для выходных видео, если её нет
#         output_dir = os.path.dirname(output_video)
#         os.makedirs(output_dir, exist_ok=True)
#
#         video_path = self.find_video(input_video)
#         cap = cv2.VideoCapture(os.path.dirname(video_path))
#         if not cap.isOpened():
#             raise ValueError(f"Не удалось открыть видеофайл '{input_video}")
#
#         frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
#         fps = int(cap.get(cv2.CAP_PROP_FPS))
#
#         out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))
#
#         print("Начало обработки видео...")
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#
#             results = self.model(frame)
#
#             for result in results:
#                 for box in result.boxes:
#                     conf, cls = float(box.conf[0]), int(box.cls[0])
#
#                     if cls == 0 and conf >= self.confidence_threshold:
#                         x1, y1, x2, y2 = map(int, box.xyxy[0])
#                         label = f"Person {conf:.2f}"
#                         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                         cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#
#             out.write(frame)
#             cv2.imshow("YOLOv8 Detection", frame)
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 print("Обработка прервана пользователем.")
#                 break
#
#         cap.release()
#         out.release()
#         cv2.destroyAllWindows()
#         print(f"Обработка завершена. Результат сохранён в '{output_video}'.")


import os
import cv2
from ultralytics import YOLO


class PeopleDetector:
    """
    Класс для детекции людей в видео с использованием YOLOv8.
    """

    def __init__(self, model_path: str, confidence_threshold: float):
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

        print(f"Загрузка модели из: {model_path}")
        try:
            self.model = YOLO(model_path)
            print("Модель успешно загружена.")
        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}")
            raise
        self.confidence_threshold = confidence_threshold

    @staticmethod
    def find_video(folder_path):
        """
        Находит первый файл с расширением .mp4 в указанной папке.

        :param folder_path: Путь к папке с видео.
        :return: Полный путь к видеофайлу или None, если файл не найден.
        """
        for file in os.listdir(folder_path):
            if file.endswith(".mp4"):
                return os.path.join(folder_path, file)
        return None

    def detect(self, input_folder: str, output_video: str):
        """
        Обрабатывает видео, выполняя детекцию людей и сохраняя результат.

        :param input_folder: Путь к папке с входным видеофайлом.
        :param output_video: Путь для сохранения обработанного видео.
        """
        # Находим видеофайл в папке
        video_path = self.find_video(input_folder)
        if not video_path:
            raise FileNotFoundError(f"Видеофайл не найден в папке: {input_folder}")

        # Создаем папку для выходных видео, если её нет
        output_dir = os.path.dirname(output_video)
        os.makedirs(output_dir, exist_ok=True)

        # Открываем видеофайл
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видеофайл: {video_path}")

        # Получаем параметры видео
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Создаем VideoWriter для сохранения результата
        out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

        print("Начало обработки видео...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Детекция людей
            results = self.model(frame)
            for result in results:
                for box in result.boxes:
                    conf, cls = float(box.conf[0]), int(box.cls[0])
                    if cls == 0 and conf >= self.confidence_threshold:  # cls == 0 - это класс "человек" в YOLO
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        label = f"Person {conf:.2f}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Сохраняем кадр
            out.write(frame)

            # Показываем кадр
            cv2.imshow("YOLOv8 Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Обработка прервана пользователем.")
                break

        # Освобождаем ресурсы
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Обработка завершена. Результат сохранён в '{output_video}'.")

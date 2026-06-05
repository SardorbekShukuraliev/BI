
import os
from roboflow import Roboflow
from ultralytics import YOLO

# Этот блок обязателен для Windows, чтобы работало ускорение загрузки данных!
if __name__ == '__main__':
    
    # 1. Скачивание твоего датасета из Roboflow
    rf = Roboflow(api_key="UtAY15qdn2akYjuvJQZ6")
    project = rf.workspace("humandetection-v9nvh").project("maybe_better_detection")
    version = project.version(1)
    dataset = version.download("yolov8")
                

    # 2. Загружаем точную Medium-модель
    print("Загрузка предобученной модели YOLOv8m...")
    model = YOLO('yolov8m.pt') 

    # 3. Запуск Fine-Tuning на твоей видеокарте RTX 3050 Ti
    print("Старт дообучения модели. Логи будут выводиться для каждой эпохи:")
    results = model.train(
        data=f"{dataset.location}/data.yaml", # Путь к скачанному датасету
        epochs=50,                           # Количество эпох обучения
        imgsz=640,                           # Размер картинок
        batch=8,                             # Размер батча для твоей RTX 3050 Ti
        device=0,                            # Принудительно запускаем на GPU (0)
        workers=2,                           # Безопасное количество потоков для Windows
        name="assbi_accurate_person_detect"  # Имя папки результатов
    )

    print("Обучение завершено успешно!")
    print("Финальные веса сохранены в папку: runs/detect/assbi_accurate_person_detect/")
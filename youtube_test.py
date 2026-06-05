import cv2
import time
import yt_dlp
from ultralytics import YOLO

if __name__ == '__main__':
    # Ссылка на YouTube видео
    # youtube_url = "https://www.youtube.com/watch?v=vRIp1N78EZo" 
    # youtube_url = "https://www.youtube.com/watch?v=u8CbGedbI08" 
    youtube_url = "https://www.youtube.com/watch?v=EO_1LWqsCNE" 
    
    # Загрузка твоей модели YOLOv8m
    model_path = r"C:\PDP\6\Bussiness Intelegence\finetuning\YOLOv8-HumanDetection\best.pt"
    print(f"Загрузка кастомных весов: {model_path}...")
    model = YOLO(model_path)

    print("Подключение к YouTube через yt-dlp напрямую...")
    
    # Настройки yt-dlp для выбора любого рабочего стрима (игнорируя отсутствие JS)
    ydl_opts = {
        'format': 'best[ext=mp4]/best', # Берем лучший доступный готовый формат mp4
        'quiet': True,
        'no_warnings': True
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            video_url = info['url'] # Получаем прямой URL видеопотока
        
        # Передаем прямую ссылку в OpenCV
        cap = cv2.VideoCapture(video_url)
    except Exception as e:
        print(f"\n[ОШИБКА] YouTube заблокировал запрос: {e}")
        print("Используй Вариант 2 (Локальный файл) — это на 100% надежно для защиты проекта!")
        exit()

    # Ограничение в 20 FPS
    target_fps = 20
    frame_duration = 1.0 / target_fps

    print("\n[УСПЕШНО] Видеопоток пойман! Система ASSBI запущена.")
    print("Нажмите 'Q' в окне видео для выхода.")

    while cap.isOpened():
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            print("Видеопоток завершен.")
            break

        # Детекция людей на твоей RTX 3050 Ti
        results = model.predict(frame, device=0, conf=0.25, verbose=False)
        
        # Отрисовка рамок YOLO
        annotated_frame = results[0].plot()

        # Подсчет объектов для Business Intelligence аналитики
        detected_objects = len(results[0].boxes)
        
        # Вывод информации на экран
        cv2.putText(
            annotated_frame, 
            f"ASSBI Live Analytics: {detected_objects} objects", 
            (30, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
            (0, 255, 0), 2
        )

        cv2.imshow("ASSBI Platform - Video Analytics Test", annotated_frame)

        # Контроль 20 FPS
        elapsed_time = time.time() - start_time
        sleep_time = frame_duration - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Работа скрипта завершена.")
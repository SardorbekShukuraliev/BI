import cv2
import yt_dlp
import os

# Ссылка на твой стрим Maria Channel 10
YOUTUBE_URL = "https://www.youtube.com/watch?v=vRIp1N78EZo" 
# YOUTUBE_URL = "https://www.youtube.com/watch?v=u8CbGedbI08" 
# YOUTUBE_URL = "https://www.youtube.com/watch?v=bbBGNNPu0rg" 

# Настройки yt-dlp
ydl_opts = {'format': 'best[ext=mp4]', 'noplaylist': True}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(YOUTUBE_URL, download=False)
    stream_url = info['url']

cap = cv2.VideoCapture(stream_url)
os.makedirs("dataset_frames", exist_ok=True)

# 1. Автоматически получаем FPS видеопотока
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps is None:
    fps = 30  # Если стрим не отдает FPS, ставим стандартные 30 кадров/сек

# 2. Высчитываем шаг: сколько кадров нужно пропустить для ровно 3 секунд
frames_to_skip = int(fps * 3)
print(f"Текущий FPS трансляции: {fps}")
print(f"Для интервала в 3 сек сохраняем каждый {frames_to_skip}-й кадр.")

frame_count = 0
max_frames = 300
global_frame_index = 0  # Счетчик абсолютно всех кадров из потока

print("Старт сбора данных...")

while cap.isOpened() and frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        print("Ошибка чтения потока или трансляция завершилась.")
        break
    
    # 3. Проверяем: если текущий кадр кратен нашему шагу — сохраняем его
    if global_frame_index % frames_to_skip == 0:
        cv2.imwrite(f"dataset_frames/frame_{frame_count:03d}.jpg", frame)
        print(f"Сохранен кадр {frame_count}/{max_frames} (Кадр в потоке: {global_frame_index})")
        frame_count += 1
        
    global_frame_index += 1

cap.release()
print(f"Готово! Все {max_frames} кадров сохранены с идеальным шагом в 3 секунды видео-времени.")
import os
import sqlite3
import random
import threading
import time
import math
import tkinter as tk
from tkinter import ttk, scrolledtext
from datetime import datetime, timedelta
import cv2
import yt_dlp
from PIL import Image, ImageTk

# --- ИНИЦИАЛИЗАЦИЯ И НАСТРОЙКИ ----------------------------------------
MODEL_PATH = r"C:\Djarvis\Deepseek_3.0_maybe\models\qwen2.5-3b-instruct-q4_k_m.gguf"
YOLO_MODEL_PATH = r"best.pt"
DB_NAME = r"assbi_analytics.db"

DB_SCHEMA = """
Таблица: traffic_stats
Описание: Данные о проходящих людях через две контрольные линии.
Столбцы:
- timestamp (TEXT, формат 'YYYY-MM-DD HH:MM:SS')
- line_id (INTEGER, номер линии: 1 или 2)
- count_in (INTEGER, количество человек, прошедших Внутрь / Вход)
- count_out (INTEGER, количество человек, прошедших Наружу / Выход)
"""

# ─── ГЕНЕРАТОР ТЕСТОВОЙ БД ───────────────────────────────────────────
def check_and_generate_db():
    if os.path.exists(DB_NAME): return
    print("💾 База данных не найдена. Создаю тестовый датасет...")
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS traffic_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
        line_id INTEGER NOT NULL, count_in INTEGER NOT NULL, count_out INTEGER NOT NULL
    )''')
    start_date = datetime.now() - timedelta(days=30)
    current_time = start_date
    while current_time <= datetime.now():
        ts_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
        hour = current_time.hour
        for line in [1, 2]:
            c_in = random.randint(10, 50) if 9 <= hour <= 21 else random.randint(0, 4)
            c_out = random.randint(8, 45) if 9 <= hour <= 21 else random.randint(0, 3)
            cursor.execute('INSERT INTO traffic_stats (timestamp, line_id, count_in, count_out) VALUES (?, ?, ?, ?)', 
                           (ts_str, line, c_in, c_out))
        current_time += timedelta(hours=1)
    conn.commit()
    conn.close()
    print("✅ База данных готова!")

# ─── КЛАСС ДАШБОРДА ASSBI PRO ────────────────────────────────────────
class ASSBIProDashboard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ASSBI PRO — Симулятор Контроля Камер, YOLOv8 & Локальный GGUF Аналитик")
        self.geometry("1430x850")
        self.configure(bg="#0f0f12")

        self.ui_ready = False

        # Размеры виртуального монитора
        self.screen_width, self.screen_height = 540, 360
        
        # Начальная геометрия Линии 1 (Центр X, Центр Y, Угол, Длина)
        self.line1_x, self.line1_y, self.line1_angle, self.line1_len = 270, 120, 0, 350
        # Начальная геометрия Линии 2 (Центр X, Центр Y, Угол, Длина)
        self.line2_x, self.line2_y, self.line2_angle, self.line2_len = 270, 260, 0, 350

        self.cap = None
        self.video_playing = False
        self.current_frame = None
        
        self.llm = None
        self.yolo_model = None
        self.yolo_active = False

        self.setup_styles()
        self.setup_ui()
        
        # Фоновая загрузка моделей
        threading.Thread(target=self.init_gguf_model, daemon=True).start()
        threading.Thread(target=self.init_yolo_model, daemon=True).start()
        
        self.ui_ready = True
        self.update_video_feed()

    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure(".", background="#16161a", foreground="#ffffff")
        self.style.configure("TScale", background="#16161a", troughcolor="#232329")

    def setup_ui(self):
        # ==================================================================
        # ⬅️ ЛЕВАЯ ПАНЕЛЬ: АНАЛИТИЧЕСКИЙ ЧАТ-БОТ
        # ==================================================================
        left_frame = tk.Frame(self, bg="#16161a", padx=15, pady=15, width=460)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH)
        left_frame.pack_propagate(False)

        chat_header = tk.Frame(left_frame, bg="#16161a")
        chat_header.pack(fill=tk.X, pady=(0, 10))
        tk.Label(chat_header, text="🧠 ASSBI LOCAL GGUF АГЕНТ", bg="#16161a", fg="#00ffcc", font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT)
        self.status_label = tk.Label(chat_header, text="• ЗАГРУЗКА ИИ...", bg="#16161a", fg="#ffaa00", font=("Segoe UI", 9, "bold"))
        self.status_label.pack(side=tk.LEFT, padx=10, pady=3)

        self.chat_area = scrolledtext.ScrolledText(
            left_frame, bg="#1e1e24", fg="#f8f8f2", insertbackground="white",
            font=("Consolas", 10), relief=tk.FLAT, highlightthickness=1, highlightbackground="#2d2d38"
        )
        self.chat_area.pack(fill=tk.BOTH, expand=True, pady=(0, 12))
        self.log_message("Система ASSBI Pro инициализируется...")
        self.chat_area.configure(state=tk.DISABLED)

        input_container = tk.Frame(left_frame, bg="#16161a")
        input_container.pack(fill=tk.X, side=tk.BOTTOM)

        self.query_entry = tk.Entry(
            input_container, bg="#1e1e24", fg="#ffffff", insertbackground="white",
            font=("Segoe UI", 11), relief=tk.FLAT, highlightthickness=1, highlightbackground="#2d2d38"
        )
        self.query_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0, 8))
        self.query_entry.bind("<Return>", lambda event: self.start_ai_agent_thread())

        self.send_btn = tk.Button(
            input_container, text="СПРОСИТЬ", bg="#00ffcc", fg="#0f0f12",
            font=("Segoe UI", 10, "bold"), relief=tk.FLAT, activebackground="#00ccaa", padx=15, cursor="hand2"
        )
        self.send_btn.pack(side=tk.RIGHT, ipady=6)
        self.send_btn.config(command=self.start_ai_agent_thread)

        # ==================================================================
        # ➡️ ПРАВАЯ ПАНЕЛЬ: МОНИТОР LIVE КАМЕРЫ И УЛУЧШЕННАЯ МАТРИЦА СЛАЙДЕРОВ
        # ==================================================================
        right_frame = tk.Frame(self, bg="#0f0f12", padx=15, pady=15)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        stream_panel = tk.Frame(right_frame, bg="#0f0f12")
        stream_panel.pack(fill=tk.X, pady=(0, 10))
        tk.Label(stream_panel, text="URL трансляции YouTube:", bg="#0f0f12", fg="#ffffff", font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=(0, 10))
        
        self.url_entry = tk.Entry(stream_panel, bg="#16161a", fg="#ffffff", insertbackground="white", relief=tk.FLAT, highlightthickness=1, highlightbackground="#2d2d38")
        self.url_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4, padx=(0, 10))
        self.url_entry.insert(0, "https://www.youtube.com/watch?v=5SIWsxZCA-E")

        self.connect_btn = tk.Button(stream_panel, text="ПОДКЛЮЧИТЬ ПОТОК", bg="#ff4444", fg="#ffffff", font=("Segoe UI", 9, "bold"), relief=tk.FLAT, command=self.start_video_thread, cursor="hand2")
        self.connect_btn.pack(side=tk.RIGHT, ipady=2)

        self.canvas_frame = tk.Frame(right_frame, bg="#16161a", bd=0, highlightthickness=1, highlightbackground="#2d2d38")
        self.canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 15))
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="#111113", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        controls_frame = tk.LabelFrame(
            right_frame, text=" 🎛️ 2D-Конфигуратор позиционирования, углов и длины линий (Управление: Мышь / Клавиатура ← →) ", 
            bg="#16161a", fg="#a0a0aa", font=("Segoe UI", 10), labelanchor="nw", padx=15, pady=10, bd=1, relief=tk.SOLID
        )
        controls_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Локальный хелпер для привязки стрелочек клавиатуры
        def bind_keyboard_arrows(scale_widget, step=2):
            scale_widget.bind("<Left>", lambda e: [scale_widget.set(scale_widget.get() - step), self.on_param_change()])
            scale_widget.bind("<Right>", lambda e: [scale_widget.set(scale_widget.get() + step), self.on_param_change()])

        # --- Ряд 0: Габариты мониторинга ---
        tk.Label(controls_frame, text="Ширина кадра:", bg="#16161a", fg="#ffffff").grid(row=0, column=0, sticky="w", pady=3)
        self.w_slider = ttk.Scale(controls_frame, from_=350, to=650, orient=tk.HORIZONTAL, command=self.on_param_change)
        self.w_slider.set(self.screen_width)
        self.w_slider.grid(row=0, column=1, padx=8, sticky="ew")
        bind_keyboard_arrows(self.w_slider, step=5)

        tk.Label(controls_frame, text="Высота кадра:", bg="#16161a", fg="#ffffff").grid(row=0, column=2, sticky="w", pady=3)
        self.h_slider = ttk.Scale(controls_frame, from_=200, to=420, orient=tk.HORIZONTAL, command=self.on_param_change)
        self.h_slider.set(self.screen_height)
        self.h_slider.grid(row=0, column=3, padx=8, sticky="ew")
        bind_keyboard_arrows(self.h_slider, step=5)

        # --- Ряд 1: ЛИНИЯ 1 (Полная настройка положения) ---
        tk.Label(controls_frame, text="Л1 Y (Высота):", bg="#16161a", fg="#ff4444", font=("Segoe UI", 9, "bold")).grid(row=1, column=0, sticky="w", pady=4)
        self.l1_slider = ttk.Scale(controls_frame, from_=0, to=420, orient=tk.HORIZONTAL, command=self.on_param_change)
        self.l1_slider.set(self.line1_y)
        self.l1_slider.grid(row=1, column=1, padx=8, sticky="ew")
        bind_keyboard_arrows(self.l1_slider, step=3)

        tk.Label(controls_frame, text="Л1 X (Смещение):", bg="#16161a", fg="#ff4444", font=("Segoe UI", 9, "bold")).grid(row=1, column=2, sticky="w", pady=4)
        self.l1_x_slider = ttk.Scale(controls_frame, from_=0, to=650, orient=tk.HORIZONTAL, command=self.on_param_change)
        self.l1_x_slider.set(self.line1_x)
        self.l1_x_slider.grid(row=1, column=3, padx=8, sticky="ew")
        bind_keyboard_arrows(self.l1_x_slider, step=3)

        tk.Label(controls_frame, text="Угол Л1 (°):", bg="#16161a", fg="#ff4444").grid(row=1, column=4, sticky="w", pady=4)
        self.l1_angle_slider = ttk.Scale(controls_frame, from_=-90, to=90, orient=tk.HORIZONTAL, command=self.on_param_change)
        self.l1_angle_slider.set(self.line1_angle)
        self.l1_angle_slider.grid(row=1, column=5, padx=8, sticky="ew")
        bind_keyboard_arrows(self.l1_angle_slider, step=1)

        tk.Label(controls_frame, text="Длина Л1 (px):", bg="#16161a", fg="#ff4444").grid(row=1, column=6, sticky="w", pady=4)
        self.l1_len_slider = ttk.Scale(controls_frame, from_=10, to=650, orient=tk.HORIZONTAL, command=self.on_param_change)
        self.l1_len_slider.set(self.line1_len)
        self.l1_len_slider.grid(row=1, column=7, padx=8, sticky="ew")
        bind_keyboard_arrows(self.l1_len_slider, step=5)

        # --- Ряд 2: ЛИНИЯ 2 (Полная настройка положения) ---
        tk.Label(controls_frame, text="Л2 Y (Высота):", bg="#16161a", fg="#3366ff", font=("Segoe UI", 9, "bold")).grid(row=2, column=0, sticky="w", pady=4)
        self.l2_slider = ttk.Scale(controls_frame, from_=0, to=420, orient=tk.HORIZONTAL, command=self.on_param_change)
        self.l2_slider.set(self.line2_y)
        self.l2_slider.grid(row=2, column=1, padx=8, sticky="ew")
        bind_keyboard_arrows(self.l2_slider, step=3)

        tk.Label(controls_frame, text="Л2 X (Смещение):", bg="#16161a", fg="#3366ff", font=("Segoe UI", 9, "bold")).grid(row=2, column=2, sticky="w", pady=4)
        self.l2_x_slider = ttk.Scale(controls_frame, from_=0, to=650, orient=tk.HORIZONTAL, command=self.on_param_change)
        self.l2_x_slider.set(self.line2_x)
        self.l2_x_slider.grid(row=2, column=3, padx=8, sticky="ew")
        bind_keyboard_arrows(self.l2_x_slider, step=3)

        tk.Label(controls_frame, text="Угол Л2 (°):", bg="#16161a", fg="#3366ff").grid(row=2, column=4, sticky="w", pady=4)
        self.l2_angle_slider = ttk.Scale(controls_frame, from_=-90, to=90, orient=tk.HORIZONTAL, command=self.on_param_change)
        self.l2_angle_slider.set(self.line2_angle)
        self.l2_angle_slider.grid(row=2, column=5, padx=8, sticky="ew")
        bind_keyboard_arrows(self.l2_angle_slider, step=1)

        tk.Label(controls_frame, text="Длина Л2 (px):", bg="#16161a", fg="#3366ff").grid(row=2, column=6, sticky="w", pady=4)
        self.l2_len_slider = ttk.Scale(controls_frame, from_=10, to=650, orient=tk.HORIZONTAL, command=self.on_param_change)
        self.l2_len_slider.set(self.line2_len)
        self.l2_len_slider.grid(row=2, column=7, padx=8, sticky="ew")
        bind_keyboard_arrows(self.l2_len_slider, step=5)

        for col in [1, 3, 5, 7]:
            controls_frame.columnconfigure(col, weight=1)

    # ─── ИНИЦИАЛИЗАЦИЯ НЕЙРОСЕТЕЙ ────────────────────────────────────────
    def init_yolo_model(self):
        try:
            from ultralytics import YOLO
            model_file = YOLO_MODEL_PATH if os.path.exists(YOLO_MODEL_PATH) else "yolov8n.pt"
            self.log_message(f"⚙️ Загрузка модели детекции людей ({model_file})...")
            self.yolo_model = YOLO(model_file)
            self.yolo_active = True
            self.log_message(f"✓ YOLOv8 ({model_file}) успешно подключена к видеопотоку!")
        except Exception as e:
            self.log_message(f"❌ Ошибка загрузки YOLOv8: {str(e)}")

    def init_gguf_model(self):
        try:
            from llama_cpp import Llama
            if os.path.exists(MODEL_PATH):
                self.llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_gpu_layers=-1, verbose=False)
                self.status_label.config(text="• РАБОТАЕТ (GGUF)", fg="#39ff14")
                self.log_message("✓ Локальная GGUF модель успешно загружена в видеопамять!")
            else:
                self.status_label.config(text="• ОШИБКА МОДЕЛИ", fg="#ff3333")
                self.log_message(f"⚠️ Файл GGUF модели не найден по пути: {MODEL_PATH}")
        except Exception as e:
            self.status_label.config(text="• ОШИБКА ИИ", fg="#ff3333")
            self.log_message(f"❌ Ошибка развертывания LLM: {str(e)}")

    # ─── ПОДКЛЮЧЕНИЕ СТРИМА YOUTUBE ──────────────────────────────────────
    def start_video_thread(self):
        url = self.url_entry.get().strip()
        if not url: return
        self.connect_btn.config(text="ПОДКЛЮЧЕНИЕ...", state=tk.DISABLED)
        threading.Thread(target=self.load_youtube_stream, args=(url,), daemon=True).start()

    def load_youtube_stream(self, url):
        try:
            self.video_playing = False
            if self.cap: self.cap.release()
            
            ydl_opts = {'format': 'best', 'quiet': True, 'no_warnings': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                stream_url = info['url']
            
            self.cap = cv2.VideoCapture(stream_url)
            self.video_playing = True
            self.log_message("✓ YouTube Live Stream успешно захвачен.")
            self.connect_btn.config(text="ПОДКЛЮЧЕНО", state=tk.NORMAL)
        except Exception as e:
            self.log_message(f"❌ Ошибка подключения стрима: {str(e)}")
            self.connect_btn.config(text="ПОВТОРИТЬ", state=tk.NORMAL)

    # ─── ОБРАБОТКА ИЗМЕНЕНИЙ СЛАЙДЕРОВ ───────────────────────────────────
    def on_param_change(self, event=None):
        if not getattr(self, 'ui_ready', False):
            return

        self.screen_width = int(self.w_slider.get())
        self.screen_height = int(self.h_slider.get())
        
        # Считываем X и Y позиции для обеих линий
        self.line1_y = int(self.l1_slider.get())
        self.line1_x = int(self.l1_x_slider.get())
        self.line1_angle = int(self.l1_angle_slider.get())
        self.line1_len = int(self.l1_len_slider.get())
        
        self.line2_y = int(self.l2_slider.get())
        self.line2_x = int(self.l2_x_slider.get())
        self.line2_angle = int(self.l2_angle_slider.get())
        self.line2_len = int(self.l2_len_slider.get())
        
        if not self.video_playing:
            self.redraw_interface_only()

    # Считаем точные 2D координаты с учетом кастомного смещения X и Y центра отрезка
    def calculate_line_coords(self, cx, cy, line_x, line_y, angle_deg, line_len):
        mid_x = cx + line_x
        mid_y = cy + line_y
        
        angle_rad = math.radians(angle_deg)
        half_len = line_len // 2

        x1 = mid_x - half_len * math.cos(angle_rad)
        y1 = mid_y - half_len * math.sin(angle_rad)
        x2 = mid_x + half_len * math.cos(angle_rad)
        y2 = mid_y + half_len * math.sin(angle_rad)
        
        return x1, y1, x2, y2

    # ─── ОБРАБОТКА КАДРОВ И ОТРИСОВКА В ПОТОКЕ ───────────────────────────
    def update_video_feed(self):
        if self.video_playing and self.cap:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (self.screen_width, self.screen_height))
                
                detected_count = 0
                if self.yolo_active and self.yolo_model:
                    results = self.yolo_model(frame, conf=0.35, verbose=False)
                    for box in results[0].boxes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        conf = float(box.conf[0].cpu().numpy())
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"Human: {conf:.2f}", (x1, y1 - 7), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                        detected_count += 1
                
                status_text = f"YOLOv8: ACTIVE | DETECTED: {detected_count}" if self.yolo_active else "YOLOv8: OFF"
                cv2.putText(frame, status_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 2)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.current_frame = ImageTk.PhotoImage(image=Image.fromarray(frame))
                
                self.draw_all_layers(has_video=True)
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        self.after(25, self.update_video_feed)

    def draw_all_layers(self, has_video=False):
        self.canvas.delete("all")
        canvas_w = max(self.canvas.winfo_width(), 700)
        canvas_h = max(self.canvas.winfo_height(), 430)

        cx = (canvas_w // 2) - (self.screen_width // 2)
        cy = (canvas_h // 2) - (self.screen_height // 2)
        x2 = cx + self.screen_width
        y2 = cy + self.screen_height

        if has_video and self.current_frame:
            self.canvas.create_image(cx, cy, image=self.current_frame, anchor="nw")
        else:
            self.canvas.create_rectangle(cx, cy, x2, y2, fill="#1c1c22", outline="#2d2d38")
            for i in range(cx + 40, x2, 40): self.canvas.create_line(i, cy, i, y2, fill="#25252d")
            for j in range(cy + 40, y2, 40): self.canvas.create_line(cx, j, x2, j, fill="#25252d")

        self.canvas.create_text(cx + 12, cy + 15, text=f"STREAM: {self.screen_width}x{self.screen_height}", fill="#00ffcc", font=("Consolas", 8), anchor="w")

        # Рисуем Линию 1 с учетом её индивидуальных X и Y координат
        l1_x1, l1_y1, l1_x2, l1_y2 = self.calculate_line_coords(cx, cy, self.line1_x, self.line1_y, self.line1_angle, self.line1_len)
        self.canvas.create_line(l1_x1, l1_y1, l1_x2, l1_y2, fill="#ff4444", width=2, dash=(6, 3))
        self.canvas.create_text(l1_x1 + 10, l1_y1 - 12, text=f"[LINE 01] X:{self.line1_x} Y:{self.line1_y} | ANG: {self.line1_angle}°", fill="#ff4444", font=("Segoe UI", 8, "bold"), anchor="w")

        # Рисуем Линию 2 с учетом её индивидуальных X и Y координат
        l2_x1, l2_y1, l2_x2, l2_y2 = self.calculate_line_coords(cx, cy, self.line2_x, self.line2_y, self.line2_angle, self.line2_len)
        self.canvas.create_line(l2_x1, l2_y1, l2_x2, l2_y2, fill="#3366ff", width=2, dash=(6, 3))
        self.canvas.create_text(l2_x1 + 10, l2_y1 - 12, text=f"[LINE 02] X:{self.line2_x} Y:{self.line2_y} | ANG: {self.line2_angle}°", fill="#3366ff", font=("Segoe UI", 8, "bold"), anchor="w")

    def redraw_interface_only(self):
        self.draw_all_layers(has_video=False)

    # ─── СИСТЕМА ИНТЕЛЛЕКТУАЛЬНОЙ БИЗНЕС-АНАЛИТИКИ (ASSBI BI-AGENT) ───────
    def log_message(self, message):
        self.chat_area.configure(state=tk.NORMAL)
        self.chat_area.insert(tk.END, message + "\n")
        self.chat_area.see(tk.END)
        self.chat_area.configure(state=tk.DISABLED)

    def start_ai_agent_thread(self):
        question = self.query_entry.get().strip()
        if not question or not self.llm: return
        self.query_entry.delete(0, tk.END)
        self.log_message(f"👤 Вы: {question}")
        threading.Thread(target=self.run_gguf_bi_agent, args=(question,), daemon=True).start()

    def run_gguf_bi_agent(self, user_question):
        current_date_mock = datetime.now().strftime("%Y-%m-%d")
        
        # Шаг 1: Превращаем человеческий вопрос в строгий SQL-запрос
        sql_prompt = f"""<|im_start|>system
Ты — ИИ-генератор строгого SQL кода для базы данных SQLite системы мониторинга ASSBI.
Схема таблицы: {DB_SCHEMA}
Текущая дата: {current_date_mock}.

Твоя задача — сгенерировать только один валидный SELECT запрос на основе вопроса пользователя. 
ВАЖНО: Выведи ТОЛЬКО чистый SQL код. Без кавычек ```sql, без объяснений, без лишних символов. Запрос должен начинаться со слова SELECT.
<|im_end|>
<|im_start|>user
Вопрос: {user_question}
<|im_end|>
<|im_start|>assistant\n"""
        
        self.log_message("🤖 Агент: Извлекаю структуру данных через GGUF...")
        
        try:
            response = self.llm(sql_prompt, max_tokens=150, stop=["<|im_end|>", "User:", "\n\n"])
            raw_sql = response['choices'][0]['text'].strip()
            raw_sql = raw_sql.replace("```sql", "").replace("```", "").strip()
            
            self.log_message(f"⚙️ Выполняю инференс SQL: {raw_sql}")
            
            # Шаг 2: Исполняем SQL запрос в локальной БД
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            cursor.execute(raw_sql)
            db_result = cursor.fetchall()
            conn.close()
            
            self.log_message(f"💾 База вернула сырые данные: {db_result}")
            
            # Шаг 3: Передаем результаты в LLM для формирования крутого бизнес-ответа
            final_prompt = f"""<|im_start|>system
Ты — ведущий бизнес-аналитик интеллектуальной платформы видеонаблюдения и подсчета трафика ASSBI (Advanced Smart Surveillance & Business Intelligence).
Твоя задача — изучить вопрос пользователя и сырые данные, полученные из SQL-базы данных, проанализировать их и дать точный, профессиональный и понятный ответ.

Схема колонок, которые вернула база: timestamp, line_id, count_in, count_out.
Если данные пустые или их нет, вежливо скажи об этом и предложи переформулировать вопрос.
Если в ответе есть цифры, сделай краткие выводы (например: посчитай общую сумму, выдели пиковый день/час или сравни линии между собой). Отвечай строго на русском языке.
<|im_end|>
<|im_start|>user
Вопрос пользователя: {user_question}
Выполненный SQL-запрос: {raw_sql}
Данные из базы данных: {db_result}
<|im_end|>
<|im_start|>assistant\n"""
            
            response_final = self.llm(final_prompt, max_tokens=350, stop=["<|im_end|>", "<|im_start|>"])
            final_answer = response_final['choices'][0]['text'].strip()
            
            self.log_message(f"✨ Агент:\n{final_answer}\n")
            
        except Exception as e:
            self.log_message(f"❌ Сбой инференса или выполнения SQL-аналитики: {str(e)}")

# ─── ТОЧКА ВХОДА В СИСТЕМУ ASSBI PRO ─────────────────────────────────
if __name__ == "__main__":
    check_and_generate_db()
    app = ASSBIProDashboard()
    app.update()
    app.redraw_interface_only()
    app.mainloop()
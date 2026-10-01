"""Giao diện phân tích giá bất động sản."""
from pathlib import Path
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText

import pandas as pd
from PIL import Image, ImageTk

BASE = Path(__file__).resolve().parent
PARTS = [
    ('Làm sạch dữ liệu', 'Part1_cleaning.py'),
    ('Khảo sát dữ liệu', 'Part2_eda.py'),
    ('Trực quan hóa', 'Part3_visualization.py'),
    ('Huấn luyện mô hình', 'Part4_modeling.py'),
    ('Nhận xét kết quả', 'Part5_insights.py'),
]
CHARTS_3 = [
    ('Giá trung bình theo quận/huyện', 'bieu_do_1_gia_trung_binh_theo_quan_huyen.png'),
    ('Phân bố giá bất động sản', 'bieu_do_2_phan_bo_gia_bat_dong_san.png'),
    ('Diện tích và giá', 'bieu_do_3_dien_tich_va_gia.png'),
    ('Giá theo loại bất động sản', 'bieu_do_4_gia_theo_loai_bat_dong_san.png'),
    ('Tương quan giữa các biến', 'bieu_do_5_ma_tran_tuong_quan.png'),
]
CHARTS_4 = [
    ('Giá thực tế và dự đoán', '01_actual_vs_predicted.png'),
    ('Phân tích phần dư', '02_residuals.png'),
    ('Sai số phần trăm', '03_percentage_errors.png'),
    ('Mức độ quan trọng của đặc trưng', '04_feature_importance.png'),
    ('So sánh dự đoán hai mô hình', 'part4_so_sanh_du_doan.png'),
    ('So sánh MAPE', 'part4_mape.png'),
    ('So sánh R²', 'part4_r2.png'),
    ('Giá thực tế và dự đoán — tổng quan', 'part4_thuc_te_du_doan.png'),
]


def chart_path(filename):
    """Ưu tiên ảnh mới nhất trong thư mục của chương trình."""
    candidates = [BASE / 'Anh' / filename,
                  BASE / 'results_modeling' / 'charts' / filename,
                  BASE / 'charts' / filename]
    existing = [path for path in candidates if path.is_file()]
    return max(existing, key=lambda path: path.stat().st_mtime) if existing else None


class ChartPanel(ttk.Frame):
    def __init__(self, parent, charts):
        super().__init__(parent, padding=12)
        self.charts = charts
        self.original = None
        self.photo = None
        self.resize_job = None
        toolbar = ttk.Frame(self)
        toolbar.pack(fill='x', pady=(0, 10))
        self.choice = ttk.Combobox(toolbar, state='readonly', width=44,
                                   values=[title for title, _ in charts])
        self.choice.pack(side='left', fill='x', expand=True, padx=(0, 8))
        self.choice.current(0)
        self.choice.bind('<<ComboboxSelected>>', self.load)
        ttk.Button(toolbar, text='Làm mới', command=self.load).pack(side='left', padx=4)
        ttk.Button(toolbar, text='Xem ảnh lớn', command=self.open_full).pack(side='left')
        self.canvas = tk.Canvas(self, bg='white', highlightthickness=1,
                                highlightbackground='#dbe3ed')
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<Configure>', self.schedule_render)
        self.note = ttk.Label(self, style='Muted.TLabel', wraplength=850)
        self.note.pack(fill='x', pady=(8, 0))
        self.load()

    def load(self, event=None):
        index = max(0, self.choice.current())
        title, filename = self.charts[index]
        path = chart_path(filename)
        self.original = None
        try:
            if path:
                with Image.open(path) as source:
                    self.original = source.convert('RGB')
                self.note.config(text=f'{title}  •  {filename}')
            else:
                self.note.config(text=f'Chưa có {filename}. Chạy bước Trực quan hóa hoặc Huấn luyện mô hình để tạo ảnh.')
        except (OSError, ValueError) as error:
            self.note.config(text=f'Không đọc được ảnh: {error}')
        self.render()

    def schedule_render(self, event=None):
        if self.resize_job:
            self.after_cancel(self.resize_job)
        self.resize_job = self.after(80, self.render)

    def render(self):
        self.resize_job = None
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        self.canvas.delete('all')
        if self.original is None:
            self.canvas.create_text(width / 2, height / 2,
                text='Chưa có biểu đồ\nChạy bước tương ứng rồi bấm Làm mới.',
                fill='#64748b', font=('Segoe UI', 12), justify='center',
                width=max(100, width - 30))
            return
        if width < 30 or height < 30:
            return
        picture = self.original.copy()
        picture.thumbnail((width - 20, height - 20), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(picture)
        self.canvas.create_image(width / 2, height / 2, image=self.photo)

    def open_full(self):
        if self.original is None:
            messagebox.showinfo('Chưa có biểu đồ', 'Hãy chạy bước tương ứng để tạo ảnh.')
            return
        window = tk.Toplevel(self)
        window.title(self.choice.get())
        window.geometry('1100x720')
        window.rowconfigure(0, weight=1)
        window.columnconfigure(0, weight=1)
        canvas = tk.Canvas(window, bg='white')
        vertical = ttk.Scrollbar(window, orient='vertical', command=canvas.yview)
        horizontal = ttk.Scrollbar(window, orient='horizontal', command=canvas.xview)
        canvas.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        canvas.grid(row=0, column=0, sticky='nsew')
        vertical.grid(row=0, column=1, sticky='ns')
        horizontal.grid(row=1, column=0, sticky='ew')
        canvas.photo = ImageTk.PhotoImage(self.original)
        canvas.create_image(0, 0, anchor='nw', image=canvas.photo)
        canvas.configure(scrollregion=canvas.bbox('all'))


class AnalysisApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Phân tích giá bất động sản Hà Nội')
        self.geometry('1240x820')
        self.minsize(920, 640)
        self.configure(bg='#f3f6fa')
        self.events = queue.Queue()
        self.busy = False
        self.buttons = []
        self.setup_style()
        header = ttk.Frame(self, padding=(24, 16))
        header.pack(fill='x')
        ttk.Label(header, text='Phân tích giá bất động sản', style='Title.TLabel').pack(anchor='w')
        ttk.Label(header, text='Hà Nội • Khảo sát dữ liệu và đánh giá mô hình dự đoán',
                  style='Muted.TLabel').pack(anchor='w', pady=(4, 0))
        body = ttk.Frame(self, padding=(16, 0, 16, 10))
        body.pack(fill='both', expand=True)
        sidebar = ttk.Frame(body, padding=12, width=220)
        sidebar.pack(side='left', fill='y', padx=(0, 12))
        sidebar.pack_propagate(False)
        ttk.Label(sidebar, text='CÁC BƯỚC THỰC HIỆN', style='Section.TLabel').pack(anchor='w', pady=(0, 14))
        for index, (title, _) in enumerate(PARTS):
            button = ttk.Button(sidebar, text=f'{index + 1}. {title}',
                                command=lambda i=index: self.start([i]))
            button.pack(fill='x', pady=4)
            self.buttons.append(button)
        ttk.Separator(sidebar).pack(fill='x', pady=16)
        button = ttk.Button(sidebar, text='Chạy toàn bộ quy trình', command=lambda: self.start(list(range(5))))
        button.pack(fill='x', pady=4)
        self.buttons.append(button)
        ttk.Button(sidebar, text='Tải lại kết quả', command=self.refresh).pack(fill='x', pady=4)
        ttk.Label(sidebar, text='Ảnh và kết quả đã lưu được nạp ngay khi mở ứng dụng.\n\nChạy lại mô hình khi dữ liệu thay đổi.',
                  style='Muted.TLabel', wraplength=180).pack(anchor='w', pady=18)
        self.tabs = ttk.Notebook(body)
        self.tabs.pack(side='left', fill='both', expand=True)
        self.overview = ttk.Frame(self.tabs, padding=16)
        self.tabs.add(self.overview, text='Kết quả mô hình')
        ttk.Label(self.overview, text='So sánh trên cùng tập kiểm tra', style='Section.TLabel').pack(anchor='w', pady=(0, 12))
        self.table = ttk.Treeview(self.overview, columns=('model', 'mae', 'rmse', 'mape', 'r2'),
                                  show='headings', height=3)
        for key, title, width in [('model', 'Mô hình', 190), ('mae', 'MAE (tỷ đồng)', 130),
                                  ('rmse', 'RMSE (tỷ đồng)', 130), ('mape', 'MAPE (%)', 110), ('r2', 'R²', 90)]:
            self.table.heading(key, text=title)
            self.table.column(key, width=width, minwidth=70, anchor='w' if key == 'model' else 'center')
        self.table.pack(fill='x')
        self.table.tag_configure('alternate', background='#edf3fa')
        self.model_note = ttk.Label(self.overview, wraplength=720, style='Muted.TLabel')
        self.model_note.pack(fill='x', pady=12)
        ttk.Label(self.overview, text='Nhận xét kết quả', style='Section.TLabel').pack(anchor='w', pady=(10, 8))
        self.insights = self.text_area(self.overview)
        self.charts3 = ChartPanel(self.tabs, CHARTS_3)
        self.tabs.add(self.charts3, text='Biểu đồ Part 3')
        self.charts4 = ChartPanel(self.tabs, CHARTS_4)
        self.tabs.add(self.charts4, text='Biểu đồ mô hình')
        self.data_tab = ttk.Frame(self.tabs, padding=12)
        self.tabs.add(self.data_tab, text='Dữ liệu và EDA')
        self.data_text = self.text_area(self.data_tab)
        self.log_tab = ttk.Frame(self.tabs, padding=12)
        self.tabs.add(self.log_tab, text='Nhật ký chạy')
        self.log = self.text_area(self.log_tab)
        footer = ttk.Frame(self, padding=(24, 8))
        footer.pack(fill='x')
        self.status = ttk.Label(footer, text='Sẵn sàng', style='Muted.TLabel')
        self.status.pack(side='left')
        self.progress = ttk.Progressbar(footer, mode='indeterminate', length=180)
        self.progress.pack(side='right')
        self.refresh()
        self.after(100, self.poll)
        self.protocol('WM_DELETE_WINDOW', self.close)

    def setup_style(self):
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('.', font=('Segoe UI', 10))
        style.configure('TFrame', background='#f3f6fa')
        style.configure('TLabel', background='#f3f6fa', foreground='#172b45')
        style.configure('Title.TLabel', font=('Segoe UI', 21, 'bold'))
        style.configure('Section.TLabel', font=('Segoe UI', 11, 'bold'))
        style.configure('Muted.TLabel', foreground='#52647a')
        style.configure('TButton', padding=(10, 10))
        style.configure('TNotebook.Tab', padding=(14, 10))
        style.configure('Treeview', rowheight=38, background='white', fieldbackground='white')
        style.configure('Treeview.Heading', font=('Segoe UI', 10, 'bold'), padding=8)
        style.map('TNotebook.Tab', background=[('selected', '#ffffff')], foreground=[('selected', '#155ba6')])

    def text_area(self, parent):
        widget = ScrolledText(parent, wrap='word', font=('Consolas', 10), bg='white',
                              fg='#23364c', relief='flat', padx=14, pady=12, state='disabled')
        widget.pack(fill='both', expand=True)
        return widget

    def put(self, widget, text, append=False):
        widget.config(state='normal')
        if not append:
            widget.delete('1.0', 'end')
        widget.insert('end', text)
        widget.see('end')
        widget.config(state='disabled')

    def refresh(self):
        self.charts3.load()
        self.charts4.load()
        self.table.delete(*self.table.get_children())
        path = BASE / 'model_comparison.csv'
        try:
            if not path.exists():
                self.model_note.config(text='Chưa có kết quả. Bấm Huấn luyện mô hình để chạy Part 4.')
                return
            data = pd.read_csv(path)
            for i, row in data.iterrows():
                values = (row.get('Model', row.get('model', '')), f"{row['MAE_VND'] / 1e9:,.3f}",
                          f"{row['RMSE_VND'] / 1e9:,.3f}", f"{row['MAPE_percent']:,.2f}", f"{row['R2']:.4f}")
                self.table.insert('', 'end', values=values, tags=('alternate',) if i % 2 else ())
            self.model_note.config(text='MAE, RMSE, MAPE càng thấp càng tốt. R² càng gần 1 càng tốt; R² không phải tỷ lệ dự đoán đúng.')
            summary = BASE / 'results_modeling' / 'Nhan_xet_ket_qua.txt'
            if summary.exists():
                self.put(self.insights, summary.read_text(encoding='utf-8'))
        except (OSError, ValueError, KeyError) as error:
            self.model_note.config(text=f'Không đọc được kết quả: {error}')

    def start(self, indices):
        if self.busy:
            return
        self.busy = True
        for button in self.buttons:
            button.config(state='disabled')
        self.progress.start(12)
        self.tabs.select(self.log_tab)
        self.put(self.log, 'Bắt đầu xử lý dữ liệu...\n')
        threading.Thread(target=self.worker, args=(indices,), daemon=True).start()

    def worker(self, indices):
        # Chỉ gửi sự kiện; toàn bộ cập nhật giao diện thực hiện trên luồng chính.
        last = None
        try:
            for index in indices:
                title, filename = PARTS[index]
                self.events.put(('status', f'Đang chạy: {title}'))
                result = subprocess.run([sys.executable, '-u', str(BASE / filename)], cwd=BASE,
                    env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'MPLBACKEND': 'Agg'},
                    capture_output=True, text=True, encoding='utf-8', errors='replace')
                output = result.stdout + ('\n' + result.stderr if result.stderr else '')
                self.events.put(('output', index, output))
                if result.returncode:
                    self.events.put(('error', f'Bước {title} chưa hoàn thành. Xem Nhật ký chạy để biết lỗi.'))
                    return
                last = index
            self.events.put(('success', last))
        except Exception as error:
            self.events.put(('error', str(error)))
        finally:
            self.events.put(('done',))

    def poll(self):
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            kind = event[0]
            if kind == 'status':
                self.status.config(text=event[1])
            elif kind == 'output':
                _, index, output = event
                self.put(self.log, f'\n{PARTS[index][0]}\n{output}\n', append=True)
                if index in (0, 1):
                    self.put(self.data_text, f'\n{PARTS[index][0]}\n{output}\n', append=True)
                if index == 4:
                    self.put(self.insights, output)
                if index == 2:
                    self.charts3.load()
                if index == 3:
                    self.refresh()
            elif kind == 'success':
                self.status.config(text='Hoàn thành. Kết quả đã được cập nhật.')
                target = self.charts3 if event[1] == 2 else self.data_tab if event[1] in (0, 1) else self.overview
                self.tabs.select(target)
            elif kind == 'error':
                self.status.config(text='Có lỗi khi chạy. Xem Nhật ký chạy.')
                self.tabs.select(self.log_tab)
                messagebox.showerror('Không thể hoàn thành', event[1])
            elif kind == 'done':
                self.busy = False
                self.progress.stop()
                for button in self.buttons:
                    button.config(state='normal')
        self.after(100, self.poll)

    def close(self):
        if self.busy:
            messagebox.showinfo('Đang xử lý', 'Hãy đợi bước hiện tại hoàn thành trước khi đóng ứng dụng.')
            return
        self.destroy()


if __name__ == '__main__':
    AnalysisApp().mainloop()

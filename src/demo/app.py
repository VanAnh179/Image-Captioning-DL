"""Tkinter window: pick an image to get its caption; type a Vietnamese sentence to see the 5 best images with captions."""

import tkinter as tk
from tkinter import filedialog, ttk

from PIL import Image, ImageOps, ImageTk

from src.retrieval.caption_search import SearchInputError

TOP_K = 5
PREVIEW_BOX = (320, 190)
THUMB_BOX = (215, 125)
IMAGE_TYPES = [("Ảnh", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Mọi tệp", "*.*")]
FONT = "Segoe UI"


def fit_image(path, box):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    image.thumbnail(box, Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(image)


def vi_score(score):
    return f"{score:.3f}".replace(".", ",")


class DemoApp:
    def __init__(self, root, engine, start_dir="."):
        self.root, self.engine, self.start_dir = root, engine, start_dir
        self._result_photos = []  # Tk drops images that nothing references
        root.title("Chú thích ảnh và tìm ảnh bằng tiếng Việt (KTVIC, chạy offline)")
        root.minsize(1240, 700)
        root.geometry("1280x730")
        style = ttk.Style()
        style.configure("TLabelframe.Label", font=(FONT, 11, "bold"))
        style.configure("Card.TFrame", background="#ffffff", relief="solid", borderwidth=1)
        self.status = tk.StringVar(value="Sẵn sàng. Chọn một ảnh, hoặc gõ một câu mô tả.")
        tk.Label(root, textvariable=self.status, anchor="w", font=(FONT, 10), fg="#445").pack(side="bottom", fill="x", padx=14, pady=(0, 6))
        self._build_caption_panel()
        self._build_search_panel()

    def _build_caption_panel(self):
        panel = ttk.LabelFrame(self.root, text="1. Chọn ảnh để sinh caption", padding=10)
        panel.pack(fill="x", padx=12, pady=(12, 6))
        self.preview = tk.Label(panel, text="(chưa chọn ảnh)", width=40, height=10, bg="#eef", fg="#667")
        self.preview.pack(side="left")
        right = ttk.Frame(panel)
        right.pack(side="left", fill="both", expand=True, padx=(16, 0))
        ttk.Button(right, text="Chọn ảnh…", command=self.choose_image).pack(anchor="w", pady=(0, 8))
        tk.Label(right, text="Caption do mô hình A sinh ra:", font=(FONT, 10), anchor="w").pack(fill="x")
        self.caption = tk.Label(right, text="", font=(FONT, 14, "bold"), wraplength=700, justify="left", anchor="nw", fg="#124")
        self.caption.pack(fill="x", pady=(4, 8))
        self.use_caption = ttk.Button(right, text="Dùng caption này để tìm ảnh", command=self.search_with_caption, state="disabled")
        self.use_caption.pack(anchor="w")

    def _build_search_panel(self):
        panel = ttk.LabelFrame(self.root, text="2. Gõ một câu tiếng Việt để tìm 5 ảnh phù hợp nhất", padding=10)
        panel.pack(fill="both", expand=True, padx=12, pady=6)
        bar = ttk.Frame(panel)
        bar.pack(fill="x")
        self.query = tk.StringVar()
        entry = ttk.Entry(bar, textvariable=self.query, font=(FONT, 12))
        entry.pack(side="left", fill="x", expand=True)
        entry.bind("<Return>", lambda _event: self.search())
        ttk.Button(bar, text="Tìm ảnh", command=self.search).pack(side="left", padx=(8, 0))
        self.note = tk.Label(panel, text="", anchor="w", font=(FONT, 10), fg="#a30")
        self.results = ttk.Frame(panel)
        self.results.pack(fill="both", expand=True, pady=(8, 0))
        entry.focus_set()

    def choose_image(self):
        path = filedialog.askopenfilename(title="Chọn ảnh", initialdir=self.start_dir, filetypes=IMAGE_TYPES)
        if path:
            self.show_caption(path)

    def show_caption(self, path):
        self.status.set("Đang sinh caption…")
        self.root.update_idletasks()
        try:
            photo = fit_image(path, PREVIEW_BOX)
            caption = self.engine.caption_image(path)
        except Exception as error:  # a file that is not an image, or a damaged one
            self.status.set(f"Không đọc được ảnh này: {error}")
            return
        self.preview.configure(image=photo, text="", width=photo.width(), height=photo.height())
        self.preview.image = photo
        self.caption.configure(text=caption)
        self.last_caption = caption
        self.use_caption.configure(state="normal")
        self.status.set("Đã sinh caption. Có thể bấm “Dùng caption này để tìm ảnh”.")

    def search_with_caption(self):
        self.query.set(self.last_caption)
        self.search()

    def show_note(self, text):
        self.note.configure(text=text)
        self.note.pack(fill="x", pady=(6, 0), before=self.results)

    def search(self):
        for child in self.results.winfo_children():
            child.destroy()
        self._result_photos = []
        self.note.pack_forget()
        try:
            hits = self.engine.search(self.query.get(), TOP_K)
        except SearchInputError as error:
            self.show_note(str(error))
            self.status.set("Chưa tìm được. Hãy sửa câu truy vấn.")
            return
        if not hits:
            self.show_note("Không có ảnh nào có từ trùng với câu này. Hãy thử mô tả khác.")
            self.status.set("Không có kết quả.")
            return
        if len(hits) < TOP_K:
            self.show_note(f"Chỉ có {len(hits)} ảnh có điểm khớp lớn hơn 0.")
        for column, hit in enumerate(hits):
            self._card(column, hit)
            self.results.columnconfigure(column, weight=1, uniform="card")
        self.status.set(f"Hiển thị {len(hits)} ảnh. Điểm là độ giống cosine (0–1) giữa câu bạn gõ và caption của ảnh.")

    def _card(self, column, hit):
        card = ttk.Frame(self.results, style="Card.TFrame", padding=6)
        card.grid(row=0, column=column, sticky="nsew", padx=4, pady=4)
        try:
            photo = fit_image(hit["image_path"], THUMB_BOX)
            self._result_photos.append(photo)
            tk.Label(card, image=photo, bg="#ffffff").pack()
        except OSError:
            tk.Label(card, text="(không mở được ảnh)", bg="#ffffff", fg="#a30", height=9).pack()
        tk.Label(card, text=f"Hạng {hit['rank']} · điểm {vi_score(hit['score'])} · ảnh {hit['image_id']}",
                 font=(FONT, 9, "bold"), bg="#ffffff", fg="#124", wraplength=215, justify="left").pack(anchor="w", pady=(6, 0))
        tk.Label(card, text=hit["caption"], font=(FONT, 10), bg="#ffffff", wraplength=215, justify="left").pack(anchor="w")
        tk.Label(card, text="Từ khớp: " + ", ".join(hit["matched_words"]), font=(FONT, 9), bg="#ffffff", fg="#075",
                 wraplength=215, justify="left").pack(anchor="w", pady=(4, 0))

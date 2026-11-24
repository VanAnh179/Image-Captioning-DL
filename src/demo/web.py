"""Local web page for the offline demo: upload an image to get a caption, type a sentence to get 5 images.

The server only listens on 127.0.0.1 and serves one static page, so it runs with no network. Images of the search
results are served by image ID, and only IDs that exist in the index can be requested.
"""

import io
import json
import mimetypes
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

from src.retrieval.caption_search import SearchInputError

PAGE = Path(__file__).with_name("static") / "index.html"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_SEARCH_BYTES = 16 * 1024
IMAGE_FORMATS = ("JPEG", "PNG", "WEBP")
TOP_K = 5


class WebDemo:
    """What the HTTP handler needs: the engine, a lock around the model, and the image lookup."""

    def __init__(self, engine, visual_search=None):
        self.engine = engine
        self.visual_search = visual_search
        self.model_lock = threading.Lock()
        self.search_lock = threading.Lock()
        root = engine.image_root.resolve()
        self.image_paths = {}
        for row in engine.index.records:
            path = (root / row["image_path"]).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise FileNotFoundError(f"Không thấy ảnh ID {row['image_id']} trong {root}. Kiểm tra --image-root.")
            self.image_paths[str(row["image_id"])] = path

    def caption_upload(self, data):
        if not data:
            raise ValueError("Chưa nhận được dữ liệu ảnh.")
        with self.model_lock:
            started = time.perf_counter()
            try:
                caption = self.engine.caption_image(io.BytesIO(data), formats=IMAGE_FORMATS)
            except Image.DecompressionBombError as error:
                raise ValueError("Ảnh có quá nhiều pixel. Hãy chọn ảnh có kích thước nhỏ hơn.") from error
            except OSError as error:
                raise ValueError("Không đọc được ảnh. Hãy chọn tệp JPG, PNG hoặc WEBP còn nguyên vẹn.") from error
        return {"caption": caption, "seconds": round(time.perf_counter() - started, 2)}

    def search(self, query):
        if self.visual_search is None:
            method = "tfidf"
            hits = self.engine.search(query, TOP_K)
        else:
            method = "visual"
            with self.search_lock:
                hits = self.visual_search.search(query, TOP_K)
        return {"query": query, "method": method, "hits": [
            {"rank": hit["rank"], "image_id": hit["image_id"], "score": hit["score"], "caption": hit["caption"],
             "matched_words": hit.get("matched_words", []), "image_url": f"/image/{hit['image_id']}"} for hit in hits]}


def make_handler(demo):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):  # keep the terminal quiet
            pass

        def send_bytes(self, status, body, content_type, cache=False):
            try:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass  # The browser can cancel a request when the user changes the image or query.

        def send_json(self, status, payload):
            self.send_bytes(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send_bytes(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path.startswith("/image/"):
                self.send_image(self.path[len("/image/"):])
            else:
                self.send_json(404, {"error": "Không có trang này."})

        def send_image(self, name):
            path = demo.image_paths.get(name)
            if path is None or not path.is_file():
                self.send_json(404, {"error": "Không có ảnh này trong chỉ mục."})
                return
            self.send_bytes(200, path.read_bytes(), mimetypes.guess_type(path.name)[0] or "image/jpeg", cache=True)

        def do_POST(self):
            limits = {"/api/caption": MAX_UPLOAD_BYTES, "/api/search": MAX_SEARCH_BYTES}
            limit = limits.get(self.path)
            if limit is None:
                self.send_json(404, {"error": "Không có chức năng này."})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
            except ValueError:
                length = -1
            if length < 0:
                self.send_json(400, {"error": "Dữ liệu gửi lên không hợp lệ."})
                return
            if length > limit:
                remaining = length if 0 < length <= 4 * MAX_UPLOAD_BYTES else 0
                while remaining:  # drain a moderately oversized body so the sender can still read our reply
                    chunk = self.rfile.read(min(65536, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                self.send_json(413, {"error": ("Ảnh lớn hơn 10 MB. Hãy chọn ảnh nhỏ hơn."
                                               if self.path == "/api/caption" else "Câu tìm kiếm quá dài.")})
                return
            body = self.rfile.read(length)
            try:
                if self.path == "/api/caption":
                    self.send_json(200, demo.caption_upload(body))
                else:
                    query = json.loads(body.decode("utf-8")).get("query", "")
                    self.send_json(200, demo.search(query))
            except SearchInputError as error:
                self.send_json(400, {"error": str(error), "code": error.code})
            except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):  # malformed JSON body
                self.send_json(400, {"error": "Yêu cầu không hợp lệ."})
            except ValueError as error:  # an upload that is empty, too large or not an image
                self.send_json(400, {"error": str(error)})

    return Handler


def serve(engine, host="127.0.0.1", port=8000, *, visual_search=None):
    server = ThreadingHTTPServer((host, port), make_handler(WebDemo(engine, visual_search)))
    server.daemon_threads = True
    return server

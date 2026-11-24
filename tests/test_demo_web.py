import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from src.demo.web import MAX_UPLOAD_BYTES, WebDemo, make_handler
from src.retrieval.caption_search import CaptionSearchIndex
from http.server import ThreadingHTTPServer

TRAIN = [{"image_id": 1, "captions": ["xe máy đỗ bên đường", "kệ hàng bày nhiều hàng hoá", "con mèo ngủ trên ghế",
                                      "người đàn ông đứng trên đường", "nhiều xe máy trên đường"]}]
PREDICTIONS = [(10, "xe máy đỗ bên đường"), (11, "kệ hàng bày nhiều hàng hoá"), (12, "con mèo ngủ trên ghế")]


class FakeEngine:
    def __init__(self, root):
        self.image_root = Path(root)
        candidates = [{"image_id": i, "image_path": f"val/{i}.jpg", "captions": ["x"] * 5} for i in (10, 11, 12)]
        self.index = CaptionSearchIndex.build(TRAIN, PREDICTIONS, candidates)

    def search(self, query, k=5):
        hits = self.index.search(query, k, self.image_root)
        return [{**hit, "rank": rank, "matched_words": ["kệ"]} for rank, hit in enumerate(hits, 1)]

    def caption_image(self, path, max_len=40):
        if Path(path).read_bytes().startswith(b"NOT"):
            raise OSError("cannot identify image file")
        return "một con mèo"


class WebDemoTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        (Path(self.folder.name) / "val").mkdir()
        (Path(self.folder.name) / "val" / "11.jpg").write_bytes(b"jpeg-bytes")
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(WebDemo(FakeEngine(self.folder.name))))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.folder.cleanup()

    def call(self, path, body=None, headers=None):
        request = urllib.request.Request(self.base + path, data=body, headers=headers or {})
        try:
            with urllib.request.urlopen(request) as reply:
                return reply.status, reply.read(), reply.headers
        except urllib.error.HTTPError as error:
            return error.code, error.read(), error.headers

    def post_json(self, path, payload):
        status, body, _ = self.call(path, json.dumps(payload).encode(), {"Content-Type": "application/json"})
        return status, json.loads(body)

    def test_page_and_image_are_served(self):
        status, body, headers = self.call("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers["Content-Type"])
        self.assertIn("ViCaption".encode(), body)
        status, body, _ = self.call("/image/11")
        self.assertEqual((status, body), (200, b"jpeg-bytes"))

    def test_only_indexed_images_can_be_requested(self):
        for path in ("/image/999", "/image/abc", "/image/..%2F..%2Fsecret", "/image/11/../12"):
            self.assertEqual(self.call(path)[0], 404, path)

    def test_search_returns_hits_with_caption_words_and_image_url(self):
        status, data = self.post_json("/api/search", {"query": "kệ hàng"})
        self.assertEqual(status, 200)
        self.assertEqual([hit["image_id"] for hit in data["hits"]], [11])
        hit = data["hits"][0]
        self.assertEqual((hit["rank"], hit["caption"], hit["image_url"]), (1, "kệ hàng bày nhiều hàng hoá", "/image/11"))
        self.assertGreater(hit["score"], 0)

    def test_bad_queries_get_a_message_not_a_crash(self):
        self.assertEqual(self.post_json("/api/search", {"query": "  "})[0], 400)
        status, data = self.post_json("/api/search", {"query": "zzz qqq"})
        self.assertEqual((status, data["code"]), (400, "unknown_terms"))
        self.assertEqual(self.post_json("/api/search", {"query": 5})[0], 400)
        self.assertEqual(self.call("/api/search", b"not json")[0], 400)
        self.assertEqual(self.call("/api/search", b"[1]")[0], 400)

    def test_caption_upload(self):
        status, body, _ = self.call("/api/caption", b"fake image bytes")
        data = json.loads(body)
        self.assertEqual((status, data["caption"]), (200, "một con mèo"))
        self.assertGreaterEqual(data["seconds"], 0)

    def test_caption_rejects_empty_non_image_and_oversized_uploads(self):
        status, body, _ = self.call("/api/caption", b"")
        self.assertEqual(status, 400)
        status, body, _ = self.call("/api/caption", b"NOT an image")
        self.assertEqual(status, 400)
        self.assertIn("ảnh", json.loads(body)["error"])
        status, _, _ = self.call("/api/caption", b"x" * (MAX_UPLOAD_BYTES + 2048))
        self.assertEqual(status, 413)

    def test_unknown_paths_are_404(self):
        self.assertEqual(self.call("/nothing")[0], 404)
        self.assertEqual(self.call("/api/other", b"{}")[0], 404)


if __name__ == "__main__":
    unittest.main()

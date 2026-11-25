import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from scripts.check_setup import folder_summary, package_status, read_requirements
from src.data.paths import CHECKPOINT_DIR, RAW_DIR
from src.training.run_log import tee_log


class CheckSetupTests(unittest.TestCase):
    def test_requirements_pins_are_read_and_compared(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "requirements.txt"
            path.write_text("numpy==0.0.1  # old\n# comment\nnot_a_real_package_xyz==1.0\nPyYAML\n", encoding="utf-8")
            pins = read_requirements(path)
        self.assertEqual(pins, {"numpy": "0.0.1", "not_a_real_package_xyz": "1.0"})
        states = {name: state for name, _, _, state in package_status(pins)}
        self.assertEqual(states, {"numpy": "khác phiên bản", "not_a_real_package_xyz": "thiếu"})

    def test_folder_summary_reports_what_is_present(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            found = {label: ok for label, _, ok, _ in folder_summary(root)}
            self.assertFalse(found["Dữ liệu thô"] or found["Checkpoint"])
            (root / CHECKPOINT_DIR / "A").mkdir(parents=True)
            (root / CHECKPOINT_DIR / "A" / "best.pt").write_bytes(b"x")
            for name in ("train-images", "public-test-images"):
                (root / RAW_DIR / name).mkdir(parents=True)
            for name in ("train_data.json", "test_data.json"):
                (root / RAW_DIR / name).write_text("[]", encoding="utf-8")
            found = {label: ok for label, _, ok, _ in folder_summary(root)}
            self.assertTrue(found["Dữ liệu thô"] and found["Checkpoint"])


class RunLogTests(unittest.TestCase):
    def test_tee_log_copies_output_into_the_log_and_restores_streams(self):
        import sys

        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "logs" / "run.log"
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                original = sys.stdout
                with tee_log(log):
                    print("epoch 1 xong")
                self.assertIs(sys.stdout, original)
            self.assertIn("epoch 1 xong", buffer.getvalue())
            self.assertIn("epoch 1 xong", log.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

import unittest
from pathlib import Path

import yaml

from scripts.check_setup import read_requirements

ROOT = Path(__file__).resolve().parent.parent


class EnvironmentFileTests(unittest.TestCase):
    def test_environment_yml_pins_match_requirements_txt(self):
        environment = yaml.safe_load((ROOT / "environment.yml").read_text(encoding="utf-8"))
        pip_lines = next(item["pip"] for item in environment["dependencies"] if isinstance(item, dict))
        pins = dict(line.split("==") for line in pip_lines)
        self.assertEqual(pins, read_requirements(ROOT / "requirements.txt"))

    def test_environment_yml_names_python_and_pip(self):
        environment = yaml.safe_load((ROOT / "environment.yml").read_text(encoding="utf-8"))
        plain = [item for item in environment["dependencies"] if isinstance(item, str)]
        self.assertTrue(any(item.startswith("python=") for item in plain))
        self.assertIn("pip", plain)
        self.assertEqual(environment["name"], "ktvic")


if __name__ == "__main__":
    unittest.main()

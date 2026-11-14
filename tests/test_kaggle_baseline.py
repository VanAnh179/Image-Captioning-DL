"""Check that custom model settings reach every Kaggle stage and isolated output."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_kaggle_baseline import main


class KaggleBaselineTests(unittest.TestCase):
    def test_custom_config_reaches_pilots_training_and_export(self):
        config = "configs/lstm_attention_matched.yaml"
        paths = [iter([Path("/kaggle/input/matched/project_bundle") / config]),
                 iter([Path("/kaggle/input/cache/resnet50_7x7_train.pt")]),
                 iter([Path("/kaggle/input/cache/resnet50_7x7_val.pt")])]
        with patch.dict(os.environ), \
                patch("scripts.run_kaggle_baseline.Path.rglob", side_effect=paths), \
                patch("scripts.run_kaggle_baseline.shutil.copytree") as copy, \
                patch("scripts.run_kaggle_baseline.Path.mkdir"), \
                patch("scripts.run_kaggle_baseline.Path.symlink_to"), \
                patch("scripts.run_kaggle_baseline.Path.write_text"), \
                patch("scripts.run_kaggle_baseline.os.chdir"), \
                patch("scripts.run_kaggle_baseline.run") as run, \
                patch("scripts.run_kaggle_baseline.shutil.make_archive") as archive, \
                patch("torch.cuda.is_available", return_value=True), \
                patch("torch.cuda.get_device_name", return_value="Tesla T4"), \
                patch("torch.set_num_threads"):
            main(["--config", config, "--run-name", "a_baseline_matched"])
        self.assertEqual(copy.call_args.args[0], Path("/kaggle/input/matched/project_bundle"))
        commands = [call.args[0] for call in run.call_args_list]
        pilot = next(command for command in commands if "scripts.check_training_pilot" in command)
        overfit = next(command for command in commands if "scripts.check_lstm_attention" in command)
        train = next(command for command in commands if "scripts.train_caption_model" in command)
        export = next(command for command in commands if "scripts.export_caption_results" in command)
        self.assertEqual(pilot[pilot.index("--a-config") + 1], str(Path(config)))
        for command in (overfit, train):
            self.assertEqual(command[command.index("--config") + 1], str(Path(config)))
        self.assertEqual(train[train.index("--output") + 1], str(Path("runs/a_baseline_matched")))
        self.assertEqual(export[export.index("--run-dir") + 1], str(Path("runs/a_baseline_matched")))
        self.assertEqual(archive.call_args.args[0], "/kaggle/working/a_baseline_matched_results")

    def test_invalid_output_name_is_rejected_before_accessing_kaggle_inputs(self):
        with patch("scripts.run_kaggle_baseline.Path.rglob") as glob, self.assertRaises(SystemExit):
            main(["--run-name", "../previous_run"])
        glob.assert_not_called()


if __name__ == "__main__":
    unittest.main()

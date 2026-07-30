import importlib.util
import tempfile
import unittest
from pathlib import Path

module_path = Path(__file__).with_name("model_summary.py")
spec = importlib.util.spec_from_file_location("model_summary", module_path)
model_summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_summary)


class ModelSummaryImageTests(unittest.TestCase):
    def test_creates_summary_jpg(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model_summary.jpg"
            created = model_summary.generate_model_summary_image(
                output_path=output_path,
                model_name="resnet18",
                architecture="ResNet-18",
                layers=[
                    ("Input", "224x224x3", "0"),
                    ("Conv1", "112x112x64", "7.1M"),
                    ("Layer1", "56x56x64", "0.5M"),
                    ("Layer2", "28x28x128", "1.2M"),
                    ("Layer3", "14x14x256", "2.3M"),
                    ("Layer4", "7x7x512", "4.7M"),
                    ("FC", "1000", "0.5M"),
                ],
            )
            self.assertTrue(created)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()

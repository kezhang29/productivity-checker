import threading
import unittest

import importlib.util
from pathlib import Path

module_path = Path(__file__).with_name("script.py")
spec = importlib.util.spec_from_file_location("script", module_path)
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


class ScriptCameraSelectionTests(unittest.TestCase):
    def test_get_camera_candidates_uses_requested_index_first(self):
        self.assertEqual(script.get_camera_candidates(2), [2, 0, 1, 2, 3])

    def test_get_camera_candidates_defaults_to_common_webcams(self):
        self.assertEqual(script.get_camera_candidates(None), [0, 1, 2, 3])

    def test_should_stop_capture_only_when_stop_event_is_set(self):
        self.assertFalse(script.should_stop_capture(None))
        event = threading.Event()
        self.assertFalse(script.should_stop_capture(event))
        event.set()
        self.assertTrue(script.should_stop_capture(event))


if __name__ == "__main__":
    unittest.main()

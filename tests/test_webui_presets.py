import importlib
import sys
import unittest


class WebUIPresetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._original_argv = sys.argv[:]
        sys.argv = ["webui.py"]
        cls.webui = importlib.import_module("webui")

    @classmethod
    def tearDownClass(cls):
        sys.argv = cls._original_argv

    def test_import_keeps_tts_unloaded(self):
        self.assertFalse(self.webui.tts.is_loaded())

    def test_subprocess_checkbox_is_in_default_preset_config(self):
        cfg = self.webui._default_ui_config()
        self.assertIn("use_subprocess_system", cfg["audio_generation"])
        self.assertTrue(cfg["audio_generation"]["use_subprocess_system"])

    def test_subprocess_checkbox_round_trips_through_preset_value_mapping(self):
        index = next(
            idx for idx, field in enumerate(self.webui._CONFIG_FIELDS) if field["key"] == "use_subprocess_system"
        )
        values = self.webui._ui_config_to_values({"audio_generation": {"use_subprocess_system": False}})
        self.assertFalse(values[index])


if __name__ == "__main__":
    unittest.main()

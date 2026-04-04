import os
import tempfile
import unittest

from indextts.utils.task_output_utils import (
    build_segment_output_path,
    create_task_output_layout,
    get_next_output_index,
    sanitize_output_basename,
)


class TaskOutputUtilsTests(unittest.TestCase):
    def test_sanitize_output_basename_strips_extension_and_invalid_chars(self):
        self.assertEqual("My_Name", sanitize_output_basename("  My:Name.wav  "))
        self.assertEqual("fallback", sanitize_output_basename("...", fallback="fallback"))

    def test_get_next_output_index_considers_existing_files_and_folders(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            os.makedirs(os.path.join(temp_dir, "0001"))
            with open(os.path.join(temp_dir, "0003.wav"), "w", encoding="utf-8") as handle:
                handle.write("x")
            os.makedirs(os.path.join(temp_dir, "used_audios"))

            self.assertEqual(4, get_next_output_index(temp_dir))

    def test_create_task_output_layout_creates_numbered_folder_and_segments(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            os.makedirs(os.path.join(temp_dir, "0002"))

            layout = create_task_output_layout(
                temp_dir,
                filename="custom-name.mp3",
                subtitle_mode=True,
                subtitle_extension=".vtt",
            )

            self.assertEqual("0003", layout["task_id"])
            self.assertTrue(os.path.isdir(layout["task_folder"]))
            self.assertTrue(os.path.isdir(layout["segments_dir"]))
            self.assertTrue(layout["final_wav_path"].endswith(os.path.join("0003", "custom-name.wav")))
            self.assertTrue(layout["final_mp3_path"].endswith(os.path.join("0003", "custom-name.mp3")))
            self.assertTrue(layout["metadata_path"].endswith(os.path.join("0003", "metadata.json")))
            self.assertTrue(layout["subtitle_copy_path"].endswith(os.path.join("0003", "source_subtitles.vtt")))
            self.assertEqual(
                os.path.join(layout["segments_dir"], "0007.wav"),
                build_segment_output_path(layout["segments_dir"], 7),
            )


if __name__ == "__main__":
    unittest.main()

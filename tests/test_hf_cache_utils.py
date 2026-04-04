import os
import tempfile
import unittest

from indextts.utils.hf_cache_utils import cached_file_path, cached_snapshot_dir, repo_cache_dir_name


class HFCacheUtilsTests(unittest.TestCase):
    def test_repo_cache_dir_name_matches_huggingface_layout(self):
        self.assertEqual(
            "models--facebook--w2v-bert-2.0",
            repo_cache_dir_name("facebook/w2v-bert-2.0"),
        )

    def test_cached_snapshot_dir_uses_ref_and_required_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_dir = os.path.join(temp_dir, "models--facebook--w2v-bert-2.0")
            snapshot_dir = os.path.join(repo_dir, "snapshots", "commit123")
            os.makedirs(snapshot_dir)
            os.makedirs(os.path.join(repo_dir, "refs"))

            with open(os.path.join(repo_dir, "refs", "main"), "w", encoding="utf-8") as handle:
                handle.write("commit123")
            with open(os.path.join(snapshot_dir, "config.json"), "w", encoding="utf-8") as handle:
                handle.write("{}")

            self.assertEqual(
                snapshot_dir,
                cached_snapshot_dir(
                    temp_dir,
                    "facebook/w2v-bert-2.0",
                    required_files=["config.json"],
                ),
            )
            self.assertIsNone(
                cached_snapshot_dir(
                    temp_dir,
                    "facebook/w2v-bert-2.0",
                    required_files=["missing.bin"],
                )
            )

    def test_cached_file_path_returns_expected_snapshot_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_dir = os.path.join(temp_dir, "models--funasr--campplus")
            snapshot_dir = os.path.join(repo_dir, "snapshots", "commit456")
            os.makedirs(snapshot_dir)
            os.makedirs(os.path.join(repo_dir, "refs"))

            file_path = os.path.join(snapshot_dir, "campplus_cn_common.bin")
            with open(os.path.join(repo_dir, "refs", "main"), "w", encoding="utf-8") as handle:
                handle.write("commit456")
            with open(file_path, "wb") as handle:
                handle.write(b"test")

            self.assertEqual(
                file_path,
                cached_file_path(temp_dir, "funasr/campplus", "campplus_cn_common.bin"),
            )


if __name__ == "__main__":
    unittest.main()

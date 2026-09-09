from datetime import date
from pathlib import Path
import re

import pytest

from ui.changelog_tab import CHANGELOG_ENTRIES
from ui.common import APP_VERSION


def test_changelog_is_current_unique_and_newest_first() -> None:
    versions = [version for version, _, _ in CHANGELOG_ENTRIES]
    release_dates = [date.fromisoformat(value) for _, value, _ in CHANGELOG_ENTRIES]

    assert versions[0] == f"v{APP_VERSION}"
    assert len(versions) == len(set(versions))
    assert release_dates == sorted(release_dates, reverse=True)
    assert all(markdown.strip() for _, _, markdown in CHANGELOG_ENTRIES)


PRIVATE_RELEASE_PATTERNS = (
    r"(?<!\w)[A-Za-z]:[\\/]",  # Local Windows paths, not https:// links.
    r"/(?:Users|home)/[^/]+/",  # Local user directories.
    r"(?<![\w-])V\d+\b(?!\.\d)",  # Run labels, not vX.Y or model-name suffixes.
    r"(?:qa_v\d+_|BROWSER_V\d+_VALIDATION|ADAPTER_COMPARISON_V\d+)",
    r"(?:V\d+_TRAINING_REPORT|VOICE_DECODER_ADAPTER_\d)",
    r"\b\d+[- ]recording (?:English |narration )?dataset\b",
    r"\bmeasured on (?:the|a|one|this|that)\b",
    r"\bindependent listening rated\b|\bmeasured recovery\b",
)


def _assert_public_release_text(text: str) -> None:
    for pattern in PRIVATE_RELEASE_PATTERNS:
        assert re.search(pattern, text, re.IGNORECASE) is None, pattern


@pytest.mark.parametrize(
    "version,release_date,markdown", CHANGELOG_ENTRIES,
    ids=[version for version, _, _ in CHANGELOG_ENTRIES],
)
def test_all_release_entries_exclude_personal_run_details(
    version: str, release_date: str, markdown: str,
) -> None:
    _assert_public_release_text(markdown)


def test_readme_release_summary_is_current_and_public() -> None:
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    summary = readme.split("## Quick Info", 1)[1].split("**Choose a route:**", 1)[0]
    assert f"**V{APP_VERSION} " in summary
    assert f"v{APP_VERSION} through v4.0" in readme
    _assert_public_release_text(summary)


def test_release_version_is_shared_with_adapter_metadata() -> None:
    from indextts.version import APP_VERSION as metadata_version

    assert APP_VERSION == metadata_version == "6.11"


@pytest.mark.parametrize("text", [
    "Use whisper-large-v3 for a second opinion.",
    "Restart after upgrading from v6.9 to v6.10.",
    "Read https://example.com/releases for public notes.",
    "Validation uses recordings from the selected dataset.",
])
def test_public_release_guard_allows_product_documentation(text: str) -> None:
    _assert_public_release_text(text)


@pytest.mark.parametrize("text", [
    r"Results were saved in D:\private\recordings.",
    "Results were saved in /home/example/recordings/.",
    "The V42 adapter was preferred in this comparison.",
    "See qa_v42_listening.md for personal results.",
    "See ADAPTER_COMPARISON_V41_V42.md for rankings.",
    "Measured on this recording collection.",
])
def test_public_release_guard_rejects_local_run_evidence(text: str) -> None:
    with pytest.raises(AssertionError):
        _assert_public_release_text(text)

from datetime import date

from ui.changelog_tab import CHANGELOG_ENTRIES
from ui.common import APP_VERSION


def test_changelog_is_current_unique_and_newest_first() -> None:
    versions = [version for version, _, _ in CHANGELOG_ENTRIES]
    release_dates = [date.fromisoformat(value) for _, value, _ in CHANGELOG_ENTRIES]

    assert versions[0] == f"v{APP_VERSION}"
    assert len(versions) == len(set(versions))
    assert release_dates == sorted(release_dates, reverse=True)
    assert all(markdown.strip() for _, _, markdown in CHANGELOG_ENTRIES)

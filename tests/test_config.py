"""Tests for Primo configuration defaults."""

from __future__ import annotations

from primo_mcp_server.config import PrimoConfig


def test_default_config_is_smu(monkeypatch):
    for key in (
        "PRIMO_BASE_URL",
        "PRIMO_DISCOVERY_BASE_URL",
        "PRIMO_VID",
        "PRIMO_INSTITUTION_NAME",
        "PRIMO_TAB_EVERYTHING",
        "PRIMO_TAB_CATALOGUE",
        "PRIMO_TAB_BOOKS_VIDEOS",
        "PRIMO_SCOPE_COMBINED",
        "PRIMO_SCOPE_LOCAL",
        "PRIMO_SCOPE_BOOKS_VIDEOS",
        "PRIMO_LIBRARIANS_FILE",
        "PRIMO_INLINE_LIBRARIAN_RECOMMENDATIONS",
        "PRIMO_LIBRARIAN_MIN_SCORE",
    ):
        monkeypatch.delenv(key, raising=False)

    config = PrimoConfig(_env_file=None)

    assert config.base_url == "https://search.library.smu.edu.sg/primaws/rest/pub"
    assert config.discovery_base_url is None
    assert config.vid == "65SMU_INST:SMU_NUI"
    assert config.institution_name == "SMU"
    assert config.tab_catalogue == "Catalogue"
    assert config.tab_everything == "Everything"
    assert config.tab_books_videos == "booksandvideos"
    assert config.scope_local == "MyInstitution"
    assert config.scope_combined == "MyInst_and_CI"
    assert config.scope_books_videos == "BooksVideos"
    assert config.librarians_file is None
    assert config.inline_librarian_recommendations is True
    assert config.librarian_min_score == 5.0


def test_user_agent_tracks_package_version():
    from importlib.metadata import version

    config = PrimoConfig(_env_file=None)
    assert config.user_agent == f"primo-mcp-server/{version('primo-mcp-server')}"

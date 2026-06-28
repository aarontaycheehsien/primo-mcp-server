"""Configuration for the Primo MCP server."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class PrimoConfig(BaseSettings):
    """Primo API configuration.

    Defaults are set for SMU (Singapore Management University).
    Override via environment variables with the PRIMO_ prefix,
    or via a .env file in the working directory.
    """

    model_config = SettingsConfigDict(env_prefix="PRIMO_", env_file=".env")

    # Institution-specific
    base_url: str = "https://search.library.smu.edu.sg/primaws/rest/pub"
    discovery_base_url: str | None = None
    vid: str = "65SMU_INST:SMU_NUI"
    # Institution code for the guest JWT endpoint. Derived from the part of
    # vid before the colon when not set explicitly.
    institution_code: str | None = None
    institution_name: str = "SMU"
    tab_everything: str = "Everything"
    tab_catalogue: str = "Catalogue"
    tab_books_videos: str = "booksandvideos"
    scope_combined: str = "MyInst_and_CI"
    scope_local: str = "MyInstitution"
    scope_books_videos: str = "BooksVideos"

    # Operational
    request_timeout: float = 30.0
    max_results_per_request: int = 50
    default_results: int = 10
    language: str = "en"
    user_agent: str = "primo-mcp-server/0.1.0"
    # Default for the Primo pcAvailability search parameter. When False,
    # CDI (Central Discovery Index) results are restricted to material the
    # institution has full text access to; when True the search is
    # "expanded" and includes records with no access. False is the safer
    # default for holdings-confirmation queries.
    include_unavailable: bool = False

    # Optional external JSON directory used for librarian recommendations.
    # No real profile data is bundled; local installs opt in by setting this.
    librarians_file: str | None = None
    inline_librarian_recommendations: bool = True
    librarian_min_score: float = 5.0

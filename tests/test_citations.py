"""Tests for citation formatting."""

from primo_mcp_server.citations import format_citation, invalid_style_message
from primo_mcp_server.models import PrimoRecord, SearchResponse


class TestCitations:
    def _make_article(self) -> PrimoRecord:
        return PrimoRecord(
            title="Digital Entrepreneurship in Practice",
            resource_type="article",
            creators=["Smith, John", "Jones, Mary"],
            authors_structured=["Smith, John", "Jones, Mary"],
            creation_date="2023-06-15",
            journal_title="Journal of Business Research",
            volume="150",
            issue="2",
            start_page="100",
            end_page="115",
            doi="10.1016/j.jbusres.2023.001",
            issn=["0148-2963"],
            peer_reviewed=True,
        )

    def _make_book(self) -> PrimoRecord:
        return PrimoRecord(
            title="Innovation Management",
            resource_type="book",
            creators=["Brown, Alice"],
            authors_structured=["Brown, Alice"],
            creation_date="2022",
            publisher="Oxford University Press",
            isbn=["9780198765432"],
        )

    def test_apa7_article(self):
        citation = format_citation(self._make_article(), "apa7")
        assert "Smith, J." in citation
        assert "Jones, M." in citation
        assert "(2023)" in citation
        assert "Digital Entrepreneurship" in citation
        assert "10.1016" in citation

    def test_apa7_book(self):
        citation = format_citation(self._make_book(), "apa7")
        assert "Brown, A." in citation
        assert "(2022)" in citation
        assert "Innovation Management" in citation
        assert "Oxford University Press" in citation

    def test_harvard_article(self):
        citation = format_citation(self._make_article(), "harvard")
        assert "(2023)" in citation
        assert "vol." in citation

    def test_chicago_article(self):
        citation = format_citation(self._make_article(), "chicago")
        assert "Smith" in citation
        assert "Jones" in citation

    def test_ieee_article(self):
        citation = format_citation(self._make_article(), "ieee")
        assert "J. Smith" in citation
        assert "doi:" in citation

    def test_two_authors_join_with_and_no_comma(self):
        chicago = format_citation(self._make_article(), "chicago")
        ieee = format_citation(self._make_article(), "ieee")
        assert chicago.startswith("Smith, J. and Jones, M.")
        assert ieee.startswith("J. Smith and M. Jones,")

    def test_three_authors_keep_serial_comma(self):
        record = self._make_article().model_copy(
            update={"authors_structured": ["Smith, John", "Jones, Mary", "Lee, Kim"]}
        )
        assert format_citation(record, "ieee").startswith(
            "J. Smith, M. Jones, and K. Lee,"
        )

    def test_invalid_style_message(self):
        assert invalid_style_message("APA7") is None
        assert "Invalid citation style" in invalid_style_message("mla")

    def test_vancouver_article(self):
        citation = format_citation(self._make_article(), "vancouver")
        assert "Smith J" in citation
        assert "Jones M" in citation

    def test_from_live_data(self, search_results_data):
        response = SearchResponse.from_api_response(search_results_data)
        for style in ["apa7", "harvard", "chicago", "ieee", "vancouver"]:
            citation = format_citation(response.records[0], style)
            assert len(citation) > 20
            assert response.records[0].title[:20] in citation

    def test_style_and_plural_article_type_are_normalised(self):
        record = self._make_article()
        record.resource_type = "articles"

        citation = format_citation(record, "APA7")

        assert "Journal of Business Research" in citation
        assert "Oxford University Press" not in citation

    def test_chinese_author_names_are_preserved(self):
        author = "\u88f4, \u5b9c\u7406."
        record = PrimoRecord(
            title="\u5b89\u6e90 : \u767c\u6398\u4e2d\u570b\u9769\u547d\u4e4b\u50b3\u7d71.",
            resource_type="book",
            creators=[author],
            authors_structured=[author],
            creation_date="2014",
            publisher="Hong Kong University Press",
        )

        for style in ["apa7", "harvard", "chicago", "ieee", "vancouver"]:
            citation = format_citation(record, style)
            assert author in citation
            assert "\u88f4, \u5b9c." not in citation


class TestYearWithoutJournal:
    """Regression: IEEE and Chicago dropped the year when jtitle was empty."""

    @staticmethod
    def _article_no_journal():
        return PrimoRecord(
            record_id="cdi_x", title="A Paper", resource_type="article",
            creators=["Tan, Mei Ling"], creation_date="2021",
        )

    def test_ieee_article_without_journal_keeps_year(self):
        assert "2021" in format_citation(self._article_no_journal(), "ieee")

    def test_chicago_article_without_journal_keeps_year(self):
        assert "2021" in format_citation(self._article_no_journal(), "chicago")

    def test_apa_article_without_journal_keeps_year(self):
        assert "2021" in format_citation(self._article_no_journal(), "apa7")

    def test_vancouver_article_without_journal_keeps_year(self):
        # Vancouver also only emits the year inside the journal block.
        assert "2021" in format_citation(self._article_no_journal(), "vancouver")

    def test_ieee_with_journal_unchanged(self):
        r = self._article_no_journal()
        r.journal_title = "J. Test"
        cite = format_citation(r, "ieee")
        assert "2021" in cite and "J. Test" in cite

"""Tests for Primo PNX model parsing."""

from primo_mcp_server.models import Facet, FacetValue, PrimoRecord, SearchResponse


class TestSearchResponse:
    def test_parse_search_results(self, search_results_data):
        response = SearchResponse.from_api_response(search_results_data)
        assert response.info.total > 0
        assert len(response.records) == 3

    def test_parse_empty_results(self, empty_results_data):
        response = SearchResponse.from_api_response(empty_results_data)
        assert response.info.total == 0
        assert len(response.records) == 0

    def test_total_count_compatibility_properties(self, search_results_data):
        response = SearchResponse.from_api_response(search_results_data)
        assert response.total_results == response.info.total
        assert response.total_local_results == response.info.total_local
        assert response.total_pc_results == response.info.total_pc

    def test_record_has_title(self, search_results_data):
        response = SearchResponse.from_api_response(search_results_data)
        for record in response.records:
            assert record.title != ""

    def test_record_has_creators(self, search_results_data):
        response = SearchResponse.from_api_response(search_results_data)
        for record in response.records:
            assert len(record.creators) > 0

    def test_record_has_type(self, search_results_data):
        response = SearchResponse.from_api_response(search_results_data)
        for record in response.records:
            assert record.resource_type == "article"

    def test_record_has_record_id(self, search_results_data):
        response = SearchResponse.from_api_response(search_results_data)
        for record in response.records:
            assert record.record_id != ""

    def test_peer_reviewed_detected(self, search_results_data):
        response = SearchResponse.from_api_response(search_results_data)
        # At least one record should be peer-reviewed
        assert any(r.peer_reviewed for r in response.records)


class TestPrimoRecord:
    def test_from_minimal_doc(self):
        """Test parsing with minimal/missing fields."""
        doc = {
            "pnx": {
                "display": {"title": ["Test Title"]},
                "control": {"recordid": ["test123"]},
            }
        }
        record = PrimoRecord.from_api_doc(doc)
        assert record.title == "Test Title"
        assert record.record_id == "test123"
        assert record.creators == []
        assert record.doi == ""

    def test_doi_extraction(self):
        doc = {
            "pnx": {
                "display": {
                    "title": ["Test"],
                    "identifier": ["ISSN: 1234-5678", "DOI: 10.1234/test"],
                },
                "control": {"recordid": ["test"]},
            }
        }
        record = PrimoRecord.from_api_doc(doc)
        assert record.doi == "10.1234/test"

    def test_scalar_fields_are_normalised(self):
        doc = {
            "pnx": {
                "display": {"title": "Test Title", "type": "book"},
                "control": {"recordid": "test123", "score": 42},
            }
        }
        record = PrimoRecord.from_api_doc(doc)
        assert record.title == "Test Title"
        assert record.record_id == "test123"
        assert record.score == 42.0


class TestNameCleaning:
    def test_strips_subfield_markers_from_creators(self):
        doc = {
            "pnx": {
                "display": {
                    "title": ["AI"],
                    "creator": ["Mueller, John, 1958- author.$$QMueller, John"],
                },
                "control": {"recordid": ["r1"]},
            }
        }
        record = PrimoRecord.from_api_doc(doc)
        assert record.creators == ["Mueller, John, 1958-"]
        assert all("$$" not in c for c in record.creators)

    def test_strips_relator_from_contributors(self):
        doc = {
            "pnx": {
                "display": {
                    "title": ["AI"],
                    "contributor": [
                        "Solo, Ashu M. G., editor.$$QSolo, Ashu M. G.",
                        "Arabnia, Hamid R., editor.$$QArabnia, Hamid R.",
                    ],
                },
                "control": {"recordid": ["r2"]},
            }
        }
        record = PrimoRecord.from_api_doc(doc)
        assert record.contributors == ["Solo, Ashu M. G.", "Arabnia, Hamid R."]

    def test_display_authors_prefers_structured(self):
        doc = {
            "pnx": {
                "display": {"title": ["AI"], "creator": ["Noise$$Qx"]},
                "addata": {"au": ["Clean, Author"]},
                "control": {"recordid": ["r3"]},
            }
        }
        record = PrimoRecord.from_api_doc(doc)
        assert record.display_authors == ["Clean, Author"]

    def test_display_authors_falls_back_to_addau(self):
        # Edited book: no au, no display.creator, but addau (editors) present.
        doc = {
            "pnx": {
                "display": {
                    "title": ["AI"],
                    "contributor": ["Solo, Ashu M. G., editor.$$QSolo, Ashu M. G."],
                },
                "addata": {"addau": ["Solo, Ashu M. G.", "Arabnia, Hamid R."]},
                "control": {"recordid": ["r4"]},
            }
        }
        record = PrimoRecord.from_api_doc(doc)
        assert record.authors_structured == []
        assert record.display_authors == ["Solo, Ashu M. G.", "Arabnia, Hamid R."]

    def test_unknown_author_when_no_names(self):
        doc = {"pnx": {"display": {"title": ["A journal"]}, "control": {"recordid": ["r5"]}}}
        record = PrimoRecord.from_api_doc(doc)
        assert record.display_authors == []

    def test_year_extracted_from_messy_dates(self):
        for raw, expected in [
            ("2021", "2021"),
            ("2021-03", "2021"),
            ("c1996", "1996"),
            ("[2019]", "2019"),
            ("1992-2000", "1992"),
            ("n.d.", ""),
        ]:
            doc = {
                "pnx": {
                    "display": {"title": ["T"], "creationdate": [raw]},
                    "control": {"recordid": ["r"]},
                }
            }
            assert PrimoRecord.from_api_doc(doc).year == expected, raw

    def test_preserves_chinese_metadata_and_splits_full_width_separator(self):
        title = "\u5b89\u6e90 : \u767c\u6398\u4e2d\u570b\u9769\u547d\u4e4b\u50b3\u7d71."
        author = "\u88f4, \u5b9c\u7406."
        contributor = "\u95bb, \u5c0f\u99ff."
        publisher = "\u9999\u6e2f : \u9999\u6e2f\u5927\u5b78\u51fa\u7248\u793e"
        communism = "\u5171\u7522\u4e3b\u7fa9"
        china = "\u4e2d\u570b"
        doc = {
            "pnx": {
                "display": {
                    "title": [title],
                    "creator": [f"{author}\uff1b{contributor}"],
                    "contributor": [f"{contributor}$$Q{contributor}"],
                    "publisher": [publisher],
                    "subject": [f"{communism}\uff1b{china}"],
                },
                "control": {"recordid": ["alma99317560802601"]},
            }
        }

        record = PrimoRecord.from_api_doc(doc)

        assert record.title == title
        assert record.creators == [author, contributor]
        assert record.contributors == [contributor]
        assert record.publisher == publisher
        assert record.subjects == [communism, china]

    def test_recovers_chinese_text_from_pure_pnx_subfields(self):
        title = "\u5b89\u6e90"
        author = "\u88f4, \u5b9c\u7406."
        doc = {
            "pnx": {
                "display": {
                    "title": [f"$$N{title}$$Rtitle"],
                    "creator": [f"$$N{author}$$Rauthor"],
                },
                "control": {"recordid": ["alma99317560802601"]},
            }
        }

        record = PrimoRecord.from_api_doc(doc)

        assert record.title == title
        assert record.creators == [author]


class TestFulltextAvailability:
    """Regression tests: no_fulltext must not be reported as available."""

    @staticmethod
    def _doc(fulltext):
        return {
            "pnx": {
                "control": {"recordid": ["cdi_test_1"]},
                "display": {"title": ["A title"]},
                "delivery": {"fulltext": fulltext},
            },
            "context": "PC",
        }

    def test_no_fulltext_is_not_available(self):
        record = PrimoRecord.from_api_doc(self._doc(["no_fulltext"]))
        assert record.fulltext_available is False

    def test_fulltext_is_available(self):
        record = PrimoRecord.from_api_doc(self._doc(["fulltext"]))
        assert record.fulltext_available is True

    def test_linktorsrc_is_available(self):
        record = PrimoRecord.from_api_doc(self._doc(["fulltext_linktorsrc"]))
        assert record.fulltext_available is True

    def test_fulltext_multiple_is_available(self):
        record = PrimoRecord.from_api_doc(self._doc(["fulltext_multiple"]))
        assert record.fulltext_available is True

    def test_mixed_tokens_with_positive_is_available(self):
        record = PrimoRecord.from_api_doc(
            self._doc(["no_fulltext", "fulltext_linktorsrc"])
        )
        assert record.fulltext_available is True

    def test_string_value_no_fulltext(self):
        record = PrimoRecord.from_api_doc(self._doc("no_fulltext"))
        assert record.fulltext_available is False

    def test_string_value_fulltext(self):
        record = PrimoRecord.from_api_doc(self._doc("fulltext"))
        assert record.fulltext_available is True

    def test_missing_delivery_defaults_false(self):
        record = PrimoRecord.from_api_doc(self._doc(None))
        assert record.fulltext_available is False

    def test_physical_availability_token_is_not_fulltext(self):
        # Values merged from the direct endpoint can include physical
        # availability tokens; these must not flag electronic full text.
        record = PrimoRecord.from_api_doc(self._doc(["available_in_library"]))
        assert record.fulltext_available is False


class TestDoiAndIdentifierParsing:
    """Regression tests for DOI extraction and identifier cleaning."""

    @staticmethod
    def _doc(identifier=None, addata=None):
        pnx = {
            "control": {"recordid": ["test"]},
            "display": {"title": ["Test"]},
        }
        if identifier is not None:
            pnx["display"]["identifier"] = identifier
        if addata is not None:
            pnx["addata"] = addata
        return {"pnx": pnx}

    def test_lowercase_doi_label_is_extracted_clean(self):
        # Old code detected "doi:" case-insensitively but split
        # case-sensitively, leaving the prefix in the stored DOI.
        record = PrimoRecord.from_api_doc(self._doc(["doi: 10.1234/abc"]))
        assert record.doi == "10.1234/abc"

    def test_subfield_encoded_doi_is_extracted(self):
        record = PrimoRecord.from_api_doc(
            self._doc(["$$CDOI$$V10.1007/978-981-15-1967-3"])
        )
        assert record.doi == "10.1007/978-981-15-1967-3"

    def test_addata_doi_preferred_over_display_identifier(self):
        record = PrimoRecord.from_api_doc(
            self._doc(
                ["DOI: 10.9999/display-version"],
                addata={"doi": ["10.1234/addata-version"]},
            )
        )
        assert record.doi == "10.1234/addata-version"

    def test_resolver_url_prefix_is_stripped(self):
        record = PrimoRecord.from_api_doc(
            self._doc(["DOI: https://doi.org/10.1234/x"])
        )
        assert record.doi == "10.1234/x"

    def test_bare_resolver_url_identifier_is_extracted(self):
        record = PrimoRecord.from_api_doc(
            self._doc(["https://doi.org/10.1234/bare"])
        )
        assert record.doi == "10.1234/bare"

    def test_semicolon_joined_subfield_identifiers_are_cleaned(self):
        record = PrimoRecord.from_api_doc(
            self._doc(["$$CISSN$$V1573-0565;$$COCLC$$V(OCoLC)38267175"])
        )
        assert record.identifiers == [
            "ISSN: 1573-0565",
            "OCLC: (OCoLC)38267175",
        ]
        assert record.doi == ""

    def test_doi_within_semicolon_joined_subfields(self):
        record = PrimoRecord.from_api_doc(
            self._doc(["$$CISBN$$V981-15-1967-6;$$CDOI$$V10.1007/test"])
        )
        assert record.doi == "10.1007/test"
        assert "ISBN: 981-15-1967-6" in record.identifiers

    def test_no_identifiers_gives_empty_doi(self):
        record = PrimoRecord.from_api_doc(self._doc())
        assert record.doi == ""
        assert record.identifiers == []

    def test_plain_labelled_identifiers_kept_readable(self):
        record = PrimoRecord.from_api_doc(
            self._doc(["ISBN: 9811519668", "DOI: 10.1007/978-981-15-1967-3"])
        )
        assert record.identifiers == [
            "ISBN: 9811519668",
            "DOI: 10.1007/978-981-15-1967-3",
        ]
        assert record.doi == "10.1007/978-981-15-1967-3"


class TestFacetParsing:
    def test_parses_facets_with_string_counts(self):
        data = {
            "facets": [
                {
                    "name": "rtype",
                    "values": [
                        {"value": "articles", "count": "412"},
                        {"value": "books", "count": 7},
                    ],
                }
            ]
        }
        facets = Facet.list_from_api_response(data)
        assert len(facets) == 1
        assert facets[0].name == "rtype"
        assert [(v.value, v.count) for v in facets[0].values] == [
            ("articles", 412),
            ("books", 7),
        ]

    def test_drops_nameless_empty_and_malformed_facets(self):
        data = {
            "facets": [
                {"name": "", "values": [{"value": "x", "count": 1}]},
                {"name": "topic", "values": []},
                "junk",
                {
                    "name": "lang",
                    "values": [
                        {"value": "", "count": 3},
                        "junk",
                        {"value": "eng", "count": "not-a-number"},
                    ],
                },
            ]
        }
        facets = Facet.list_from_api_response(data)
        assert [f.name for f in facets] == ["lang"]
        assert [(v.value, v.count) for v in facets[0].values] == [("eng", 0)]

    def test_missing_or_null_facets_key(self):
        assert Facet.list_from_api_response({}) == []
        assert Facet.list_from_api_response({"facets": None}) == []

    def test_search_response_defaults_to_no_facets(self, search_results_data):
        response = SearchResponse.from_api_response(search_results_data)
        assert response.facets == []



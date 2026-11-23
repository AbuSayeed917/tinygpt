from pipeline.pii import redact_pii


def test_redacts_email_phone_and_ip():
    # Arrange
    text = "Mail jane.doe@example.co.uk, call +44 20 7946 0958, server 192.168.100.200."

    # Act
    result = redact_pii(text)

    # Assert
    assert result.text == "Mail [EMAIL], call [PHONE], server [IP]."
    assert dict(result.counts) == {"email": 1, "ipv4": 1, "phone": 1}
    assert result.has_redactions


def test_leaves_years_ranges_and_short_numbers_alone():
    text = "Between 2019-2020 sales rose 12.5% to 4500 units in 3 regions."

    result = redact_pii(text)

    assert result.text == text
    assert not result.has_redactions


def test_leaves_isbns_pubmed_ids_and_page_ranges_alone():
    # Arrange: false positives seen when sampling FineWeb-Edu
    text = "ISBN 9780262038270, PMID 24462672, Revue (1968) 359-374, MET 220460258."

    # Act
    result = redact_pii(text)

    # Assert
    assert result.text == text


def test_redacts_common_phone_shapes():
    text = "Call 1-800-525-8084, (701) 328-2666 or +60126725888 today."

    result = redact_pii(text)

    assert result.text == "Call 1-[PHONE], [PHONE] or [PHONE] today."
    assert result.counts["phone"] == 3


def test_ip_is_not_mistaken_for_phone():
    result = redact_pii("10.0.0.1")

    assert result.text == "[IP]"
    assert result.counts["phone"] == 0

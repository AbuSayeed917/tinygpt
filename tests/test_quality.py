from pipeline.quality import QualityThresholds, assess_quality


def test_keeps_natural_prose(good_paragraph):
    verdict = assess_quality(good_paragraph)

    assert verdict.is_kept, verdict.failed_rules


def test_rejects_too_short_text():
    verdict = assess_quality("Buy now. Click here.")

    assert "too_few_words" in verdict.failed_rules


def test_rejects_keyword_spam_with_repeated_bigram():
    spam = " ".join(["cheap shoes"] * 60) + " the and of"

    verdict = assess_quality(spam)

    assert "repeated_bigram" in verdict.failed_rules
    assert not verdict.is_kept


def test_rejects_boilerplate_with_duplicate_lines(good_paragraph):
    boilerplate = "\n".join([good_paragraph] + ["Accept all cookies"] * 5)

    verdict = assess_quality(boilerplate)

    assert "duplicate_lines" in verdict.failed_rules


def test_rejects_numeric_tables():
    table = " ".join(str(number) for number in range(200)) + " the and"

    verdict = assess_quality(table)

    assert "non_alphabetic_words" in verdict.failed_rules


def test_empty_text_is_rejected_as_empty():
    verdict = assess_quality("   \n  ")

    assert verdict.failed_rules == ("empty",)


def test_custom_thresholds_change_the_verdict(good_paragraph):
    strict = QualityThresholds(min_words=1_000)

    verdict = assess_quality(good_paragraph, strict)

    assert verdict.failed_rules == ("too_few_words",)

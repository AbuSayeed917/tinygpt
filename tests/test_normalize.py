import pytest

from pipeline.normalize import normalize_text


def test_folds_full_width_letters_and_ligatures():
    # Arrange
    text = "ｆｉｌｅ and ﬁle"

    # Act
    result = normalize_text(text)

    # Assert
    assert result == "file and file"


def test_removes_zero_width_and_control_characters_but_keeps_tabs_and_newlines():
    text = "pass​word‮\x07\tok\nnext"

    result = normalize_text(text)

    assert result == "password\tok\nnext"


def test_collapses_blank_line_runs_and_trailing_spaces():
    text = "first   \r\n\r\n\r\n\r\nsecond\r"

    result = normalize_text(text)

    assert result == "first\n\nsecond"


def test_rejects_non_string_input():
    with pytest.raises(TypeError):
        normalize_text(b"bytes")  # type: ignore[arg-type]
